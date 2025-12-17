"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import draccus
import numpy as np
import requests
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    paraphrase_json: Optional[str] = None            # Optional: Path to JSON file with prompt paraphrases to test

    #################################################################################################################
    # Llama3 Filter Configuration (for sanitizing paraphrased/conversation prompts)
    #################################################################################################################
    use_llama_filter: bool = False                   # Whether to use Llama3 to sanitize paraphrased prompts
    llama_filter_url: str = "http://localhost:6000/filter"  # URL of the Llama3 filter server
    only_use_paraphrases: bool = False               # If True, do NOT run original prompts; only run JSON prompts

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    hf_cache_dir: Optional[str] = None               # Custom Hugging Face cache directory (overrides HF_HOME)
    resume_results_path: Optional[str] = None        # Optional: path to existing *_paraphrase_results.json to resume from
    resume_from_txt: Optional[str] = None            # Optional: path to existing .txt log file to resume from

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def parse_txt_log_for_resume(txt_log_path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Parse a .txt log file to extract trial results for resuming.
    
    Handles two cases:
    1. Actual trial execution lines: "Task: ... | Prompt: '...' | Trial X/Y" followed by "Success: True|False"
    2. Skipped prompts from previous resumes: "[ORIGINAL/PARAPHRASE] Testing prompt: '...'" followed by
       "[SKIP] Prompt already has X/Y trials completed; skipping."
    
    Returns a dict: {task_description: {prompt_variant: {"success": count, "total": count}}}
    """
    results: Dict[str, Dict[str, Dict[str, int]]] = {}
    
    with open(txt_log_path, "r") as f:
        lines = f.readlines()
    
    current_task = None
    current_prompt = None
    
    # Pattern to match: Task: <task> | Prompt: '<prompt>' | Trial X/Y
    trial_pattern = re.compile(r"Task: (.+?) \| Prompt: '(.+?)' \| Trial (\d+)/(\d+)")
    # Pattern to match: Success: True|False
    success_pattern = re.compile(r"Success: (True|False)")
    # Pattern to match: Task 'task_desc' has N paraphrases to test
    task_info_pattern = re.compile(r"Task '(.+?)' has \d+ paraphrases? to test")
    # Pattern to match: [ORIGINAL] Testing prompt: '...' or [PARAPHRASE X] Testing/Original paraphrased prompt: '...'
    testing_prompt_pattern = re.compile(r"\[(ORIGINAL|PARAPHRASE \d+)\] (?:Testing prompt|Original paraphrased prompt): '(.+?)'$")
    # Pattern to match: [SKIP] Prompt already has X/Y trials completed; skipping.
    skip_pattern = re.compile(r"\[SKIP\] Prompt already has (\d+)/(\d+) trials completed; skipping\.")
    # Patterns to match the final "PARAPHRASE RESULTS SUMMARY" section lines (if present)
    summary_task_header_pattern = re.compile(r"^Task: (.+?)$")
    summary_prompt_line_pattern = re.compile(r"^\s+\[(ORIGINAL|PARAPHRASE)\] '(.+?)': (\d+)/(\d+) \(")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse the "PARAPHRASE RESULTS SUMMARY" section if present.
        # This is the most reliable source because it includes BOTH success and total counts,
        # even if the run was itself resuming and skipped many prompts.
        summary_task_match = summary_task_header_pattern.match(line)
        if summary_task_match:
            possible_task = summary_task_match.group(1)
            # Heuristic: treat this as a summary-task header only if the next few lines look like summary entries.
            # Otherwise this might be some other "Task:" line (rare).
            found_summary_entry = False
            for j in range(i + 1, min(i + 6, len(lines))):
                if summary_prompt_line_pattern.match(lines[j]):
                    found_summary_entry = True
                    break
            if found_summary_entry:
                current_task = possible_task
                if current_task not in results:
                    results[current_task] = {}
                # Consume subsequent summary lines for this task.
                k = i + 1
                while k < len(lines):
                    entry_line = lines[k].rstrip("\n")
                    entry_match = summary_prompt_line_pattern.match(entry_line)
                    if entry_match:
                        prompt_text = entry_match.group(2)
                        success_count = int(entry_match.group(3))
                        total_count = int(entry_match.group(4))
                        if prompt_text not in results[current_task]:
                            results[current_task][prompt_text] = {"success": success_count, "total": total_count}
                        else:
                            # Prefer larger totals (and align success with that total).
                            prev_total = int(results[current_task][prompt_text].get("total", 0))
                            prev_success = int(results[current_task][prompt_text].get("success", 0))
                            if total_count > prev_total:
                                results[current_task][prompt_text] = {"success": success_count, "total": total_count}
                            elif total_count == prev_total:
                                results[current_task][prompt_text]["success"] = max(prev_success, success_count)
                        k += 1
                        continue
                    # Stop when we hit a blank line or the next task header or any non-summary content.
                    if entry_line.strip() == "" or summary_task_header_pattern.match(entry_line.strip()):
                        break
                    k += 1
                i += 1
                continue
        
        # Check for task info line (gives us the current task description)
        task_info_match = task_info_pattern.search(line)
        if task_info_match:
            current_task = task_info_match.group(1)
            i += 1
            continue
        
        # Check for trial line (actual trial execution)
        trial_match = trial_pattern.search(line)
        if trial_match:
            current_task = trial_match.group(1)
            current_prompt = trial_match.group(2)
            
            # Initialize if needed
            if current_task not in results:
                results[current_task] = {}
            if current_prompt not in results[current_task]:
                results[current_task][current_prompt] = {"success": 0, "total": 0}
            
            # Look for success line in the next few lines
            for j in range(i + 1, min(i + 10, len(lines))):
                success_match = success_pattern.search(lines[j])
                if success_match:
                    results[current_task][current_prompt]["total"] += 1
                    if success_match.group(1) == "True":
                        results[current_task][current_prompt]["success"] += 1
                    break
            i += 1
            continue
        
        # Check for "Testing prompt" lines (to detect skipped prompts from previous resumes)
        testing_match = testing_prompt_pattern.search(line)
        if testing_match:
            prompt_type = testing_match.group(1)  # "ORIGINAL" or "PARAPHRASE X"
            prompt_text = testing_match.group(2)
            
            if prompt_type == "ORIGINAL":
                # Original prompt IS the task description
                current_task = prompt_text
                current_prompt = prompt_text
            else:
                # Paraphrase - use current_task (set by "Task '...' has N paraphrases" or previous ORIGINAL)
                current_prompt = prompt_text
            
            # Check if the next non-empty line is a skip line
            for j in range(i + 1, min(i + 5, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                skip_match = skip_pattern.search(next_line)
                if skip_match and current_task is not None:
                    completed = int(skip_match.group(1))
                    # We don't know the exact success count from skip lines, but we know total
                    # For resume purposes, we need total to decide whether to skip
                    # Note: success count will be 0 (unknown) - final stats may be incomplete for these
                    if current_task not in results:
                        results[current_task] = {}
                    if current_prompt not in results[current_task]:
                        results[current_task][current_prompt] = {"success": 0, "total": completed}
                    else:
                        # Use max total in case of overlapping data
                        results[current_task][current_prompt]["total"] = max(
                            results[current_task][current_prompt]["total"],
                            completed
                        )
                break
        
        i += 1
    
    return results


def filter_instruction_remote(noisy_instruction: str, filter_url: str, log_file=None) -> str:
    """
    Send a noisy/paraphrased instruction to the Llama3 filter server for sanitization.
    
    Args:
        noisy_instruction: The original paraphrased/conversation prompt
        filter_url: URL of the Llama3 filter server
        log_file: Optional file handle for logging
        
    Returns:
        The sanitized/clean command from Llama3, or the original if filtering fails
    """
    print(f"[LLAMA3 FILTER] Input: '{noisy_instruction}'")
    if log_file:
        log_file.write(f"[LLAMA3 FILTER] Input: '{noisy_instruction}'\n")
    
    try:
        response = requests.post(
            filter_url,
            json={"instruction": noisy_instruction},
            timeout=30  # Wait max 30 seconds for Llama3 (may be slow on first request)
        )
        if response.status_code == 200:
            clean_command = response.json()['clean_command']
            print(f"[LLAMA3 FILTER] Output: '{clean_command}'")
            if log_file:
                log_file.write(f"[LLAMA3 FILTER] Output: '{clean_command}'\n")
            return clean_command
        else:
            error_msg = f"[LLAMA3 FILTER] Error {response.status_code}: Returning original instruction."
            print(error_msg)
            if log_file:
                log_file.write(error_msg + "\n")
            return noisy_instruction
    except requests.exceptions.Timeout:
        error_msg = "[LLAMA3 FILTER] Request timed out (30s). Returning original instruction."
        print(error_msg)
        if log_file:
            log_file.write(error_msg + "\n")
        return noisy_instruction
    except requests.exceptions.ConnectionError as e:
        error_msg = f"[LLAMA3 FILTER] Connection failed (Is server.py running?): {e}"
        print(error_msg)
        if log_file:
            log_file.write(error_msg + "\n")
        return noisy_instruction
    except Exception as e:
        error_msg = f"[LLAMA3 FILTER] Unexpected error: {e}"
        print(error_msg)
        if log_file:
            log_file.write(error_msg + "\n")
        return noisy_instruction


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set Hugging Face cache directory if specified
    if cfg.hf_cache_dir is not None:
        os.makedirs(cfg.hf_cache_dir, exist_ok=True)
        # Set HF_HOME to point to the cache directory
        # Hugging Face will use {HF_HOME}/hub for model cache
        os.environ["HF_HOME"] = cfg.hf_cache_dir
        print(f"[*] Using custom Hugging Face cache directory: {cfg.hf_cache_dir}")

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Load paraphrase JSON if provided
    paraphrase_dict: Dict[str, List[str]] = {}
    if cfg.paraphrase_json is not None and os.path.exists(cfg.paraphrase_json):
        with open(cfg.paraphrase_json, "r") as f:
            paraphrase_dict = json.load(f)
        print(f"Loaded {len(paraphrase_dict)} tasks with paraphrases from {cfg.paraphrase_json}")
    
    # Log Llama3 filter configuration
    if cfg.use_llama_filter:
        print(f"[*] Llama3 Filter ENABLED - URL: {cfg.llama_filter_url}")
        print("    Paraphrased/conversation prompts will be sanitized before being sent to OpenVLA")
        print("    Original prompts will be SKIPPED entirely in this mode")
    else:
        print("[*] Llama3 Filter DISABLED - Using raw prompts")

    # Optionally load previous paraphrase results to resume from (JSON or TXT)
    resume_results: Dict[str, Dict[str, Dict[str, int]]] = {}
    if cfg.resume_from_txt is not None:
        # Parse from .txt log file
        if os.path.exists(cfg.resume_from_txt):
            resume_results = parse_txt_log_for_resume(cfg.resume_from_txt)
            total_trials = sum(
                stats["total"]
                for task_prompts in resume_results.values()
                for stats in task_prompts.values()
            )
            print(f"[*] Resuming from TXT log at {cfg.resume_from_txt}")
            print(f"    Found {len(resume_results)} tasks with {total_trials} total completed trials")
        else:
            print(f"[!] Warning: resume_from_txt '{cfg.resume_from_txt}' does not exist; starting fresh.")
    elif cfg.resume_results_path is not None:
        # Load from JSON file
        if os.path.exists(cfg.resume_results_path):
            with open(cfg.resume_results_path, "r") as f:
                resume_results = json.load(f)
            print(f"[*] Resuming from existing JSON results at {cfg.resume_results_path}")
        else:
            print(f"[!] Warning: resume_results_path '{cfg.resume_results_path}' does not exist; starting fresh.")

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    # Add automatic run tags for easy distinction in filenames/logs
    if cfg.use_llama_filter:
        run_id += "--LLAMA_FILTER"
    if cfg.only_use_paraphrases and not cfg.use_llama_filter:
        run_id += "--ONLY_PARAPHRASES"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Initialize results tracking for paraphrases
    results: Dict[str, Dict[str, Dict[str, int]]] = {}  # {task_name: {prompt: {success: count, total: count}}}

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Get all prompts to test (original + paraphrases if available)
        # NOTE: If llama filtering is enabled, we intentionally do NOT run the original prompt at all.
        #       (User wants evaluation only on paraphrased / conversational prompts in that mode.)
        prompts_to_test: List[str] = []
        skip_original_prompt = cfg.use_llama_filter or cfg.only_use_paraphrases
        if not skip_original_prompt:
            prompts_to_test.append(task_description)  # Start with original (unless explicitly disabled)

        if task_description in paraphrase_dict:
            prompts_to_test.extend(paraphrase_dict[task_description])
            print(f"Task '{task_description}' has {len(paraphrase_dict[task_description])} paraphrases to test")
        elif cfg.paraphrase_json is not None:
            print(f"Warning: No paraphrases found for task '{task_description}' in {cfg.paraphrase_json}")

        # If we are skipping originals and none exist for this task, skip it to avoid silently running originals.
        if skip_original_prompt and len(prompts_to_test) == 0:
            reason = "use_llama_filter=True" if cfg.use_llama_filter else "only_use_paraphrases=True"
            print(f"[SKIP] {reason} and no prompts found for task '{task_description}'; skipping task.")
            log_file.write(
                f"[SKIP] {reason} and no prompts found for task '{task_description}'; skipping task.\n"
            )
            log_file.flush()
            continue

        # Initialize results for this task
        if task_description not in results:
            results[task_description] = {}
        for prompt in prompts_to_test:
            # If we have previous results for this task / prompt, seed from them
            if (
                task_description in resume_results
                and prompt in resume_results[task_description]
                and prompt not in results[task_description]
            ):
                prev_stats = resume_results[task_description][prompt]
                results[task_description][prompt] = {
                    "success": int(prev_stats.get("success", 0)),
                    "total": int(prev_stats.get("total", 0)),
                }
            elif prompt not in results[task_description]:
                results[task_description][prompt] = {"success": 0, "total": 0}

        # Start episodes - test each prompt variant
        task_episodes, task_successes = 0, 0
        for prompt_idx, prompt_variant in enumerate(prompts_to_test):
            is_original = prompt_variant == task_description
            prompt_label = "[ORIGINAL]" if is_original else f"[PARAPHRASE {prompt_idx}]"
            
            # Apply Llama3 filter to paraphrased/conversation prompts (not original)
            prompt_for_model = prompt_variant
            if cfg.use_llama_filter and not is_original:
                print(f"\n{prompt_label} Original paraphrased prompt: '{prompt_variant}'")
                log_file.write(f"\n{prompt_label} Original paraphrased prompt: '{prompt_variant}'\n")
                prompt_for_model = filter_instruction_remote(prompt_variant, cfg.llama_filter_url, log_file)
                print(f"{prompt_label} Filtered prompt for model: '{prompt_for_model}'")
                log_file.write(f"{prompt_label} Filtered prompt for model: '{prompt_for_model}'\n")
            else:
                print(f"\n{prompt_label} Testing prompt: '{prompt_variant}'")
                log_file.write(f"\n{prompt_label} Testing prompt: '{prompt_variant}'\n")

            # Determine how many trials have already been completed for this prompt (if resuming)
            completed_trials = results[task_description][prompt_variant]["total"]
            if completed_trials >= cfg.num_trials_per_task:
                # Nothing left to do for this prompt
                print(
                    f"  [SKIP] Prompt already has {completed_trials}/{cfg.num_trials_per_task} trials completed; skipping."
                )
                log_file.write(
                    f"  [SKIP] Prompt already has {completed_trials}/{cfg.num_trials_per_task} trials completed; skipping.\n"
                )
                continue

            for episode_idx in tqdm.tqdm(
                range(completed_trials, cfg.num_trials_per_task),
                desc=f"Prompt {prompt_idx+1}/{len(prompts_to_test)}",
            ):
                # Note: `prompt_variant` is the raw/original prompt (from JSON or original task text),
                # while `prompt_for_model` is what we actually send to OpenVLA (filtered if Llama3 filter is enabled).
                print(
                    f"\nTask: {task_description} | Prompt_raw: '{prompt_variant}' | Prompt_used: '{prompt_for_model}' "
                    f"| Trial {episode_idx + 1}/{cfg.num_trials_per_task}"
                )
                log_file.write(
                    f"\nTask: {task_description} | Prompt_raw: '{prompt_variant}' | Prompt_used: '{prompt_for_model}' "
                    f"| Trial {episode_idx + 1}/{cfg.num_trials_per_task}\n"
                )

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []
                done = False
                if cfg.task_suite_name == "libero_spatial":
                    max_steps = 220  # longest training demo has 193 steps
                elif cfg.task_suite_name == "libero_object":
                    max_steps = 280  # longest training demo has 254 steps
                elif cfg.task_suite_name == "libero_goal":
                    max_steps = 300  # longest training demo has 270 steps
                elif cfg.task_suite_name == "libero_10":
                    max_steps = 520  # longest training demo has 505 steps
                elif cfg.task_suite_name == "libero_90":
                    max_steps = 400  # longest training demo has 373 steps

                print(f"Starting episode {task_episodes+1}...")
                log_file.write(f"Starting episode {task_episodes+1}...\n")
                while t < max_steps + cfg.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue

                        # Get preprocessed image
                        img = get_libero_image(obs, resize_size)

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        # Prepare observations dict
                        # Note: OpenVLA does not take proprio state as input
                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        # Query model to get action (use filtered prompt if available)
                        action = get_action(
                            cfg,
                            model,
                            observation,
                            prompt_for_model,  # Use the filtered prompt (or original if not filtering)
                            processor=processor,
                        )

                        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                        action = normalize_gripper_action(action, binarize=True)

                        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        print(f"Caught exception: {e}")
                        log_file.write(f"Caught exception: {e}\n")
                        done = False  # Mark as failed if exception occurs
                        break

                # After episode completes, update counters and save video
                task_episodes += 1
                total_episodes += 1

                # Update results for this prompt variant
                results[task_description][prompt_variant]["total"] += 1
                if done:
                    results[task_description][prompt_variant]["success"] += 1

                # Save a replay video of the episode (only for the first trial of each prompt variant)
                if episode_idx == 0:
                    video_desc = f"{task_description[:30]}_p{prompt_idx}_t{episode_idx}"
                    if cfg.use_llama_filter:
                        video_desc = f"LLAMA_FILTER__{video_desc}"
                    save_rollout_video(
                        replay_images, total_episodes, success=done, task_description=video_desc, log_file=log_file
                    )

                # Log current results
                prompt_success_rate = results[task_description][prompt_variant]["success"] / results[task_description][prompt_variant]["total"] * 100
                print(f"Success: {done} | Prompt success rate: {prompt_success_rate:.1f}%")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                log_file.write(f"Success: {done} | Prompt success rate: {prompt_success_rate:.1f}%\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                log_file.flush()

        # Log final results (guard against division by zero when resuming)
        if task_episodes > 0:
            current_task_rate = float(task_successes) / float(task_episodes)
            print(f"Current task success rate: {current_task_rate}")
            log_file.write(f"Current task success rate: {current_task_rate}\n")
        else:
            print("Current task success rate: N/A (no new episodes run for this task)")
            log_file.write("Current task success rate: N/A (no new episodes run for this task)\n")

        if total_episodes > 0:
            current_total_rate = float(total_successes) / float(total_episodes)
            print(f"Current total success rate: {current_total_rate}")
            log_file.write(f"Current total success rate: {current_total_rate}\n")
        else:
            print("Current total success rate: N/A (no episodes run)")
            log_file.write("Current total success rate: N/A (no episodes run)\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Print and log paraphrase results summary if paraphrases were used
    if cfg.paraphrase_json is not None and len(results) > 0:
        print(f"\n{'='*80}")
        print("PARAPHRASE RESULTS SUMMARY")
        print(f"{'='*80}")
        log_file.write(f"\n{'='*80}\n")
        log_file.write("PARAPHRASE RESULTS SUMMARY\n")
        log_file.write(f"{'='*80}\n")

        original_total = 0
        original_success = 0
        paraphrase_total = 0
        paraphrase_success = 0

        for task_name, prompt_results in results.items():
            print(f"\nTask: {task_name}")
            log_file.write(f"\nTask: {task_name}\n")
            for prompt, stats in prompt_results.items():
                success_count = stats["success"]
                total_count = stats["total"]
                success_rate = (success_count / total_count * 100) if total_count > 0 else 0.0
                
                is_original = prompt == task_name
                marker = "[ORIGINAL]" if is_original else "[PARAPHRASE]"
                print(f"  {marker} '{prompt}': {success_count}/{total_count} ({success_rate:.2f}%)")
                log_file.write(f"  {marker} '{prompt}': {success_count}/{total_count} ({success_rate:.2f}%)\n")
                
                if is_original:
                    original_total += total_count
                    original_success += success_count
                else:
                    paraphrase_total += total_count
                    paraphrase_success += success_count

        original_rate = (original_success / original_total * 100) if original_total > 0 else 0.0
        paraphrase_rate = (paraphrase_success / paraphrase_total * 100) if paraphrase_total > 0 else 0.0

        print(f"\nOverall Comparison:")
        print(f"  Original Prompts: {original_success}/{original_total} ({original_rate:.2f}%)")
        print(f"  Paraphrased Prompts: {paraphrase_success}/{paraphrase_total} ({paraphrase_rate:.2f}%)")
        log_file.write(f"\nOverall Comparison:\n")
        log_file.write(f"  Original Prompts: {original_success}/{original_total} ({original_rate:.2f}%)\n")
        log_file.write(f"  Paraphrased Prompts: {paraphrase_success}/{paraphrase_total} ({paraphrase_rate:.2f}%)\n")

        # Save detailed results to JSON
        results_json_path = os.path.join(cfg.local_log_dir, run_id + "_paraphrase_results.json")
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_json_path}")
        log_file.write(f"\nDetailed results saved to: {results_json_path}\n")

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
