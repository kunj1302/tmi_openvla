# OpenVLA Fine-tuning & Evaluation Guide

This guide provides step-by-step instructions for fine-tuning OpenVLA models on LIBERO tasks and evaluating their performance, using absolute paths for all directories and files.

---

## 1. Download and Prepare the RLDS Dataset

### Download Only a Specific LIBERO Dataset in RLDS 

To download only the required dataset (e.g., `libero_goal_no_noops`), run:

```bash
huggingface-cli download openvla/modified_libero_rlds \
  --local-dir modified_libero_rlds \
  --repo-type dataset \
  --include "libero_goal_no_noops/*"
```

Replace `libero_goal_no_noops` with the dataset you need (e.g., `libero_object_no_noops`, `libero_spatial_no_noops`, etc.).

- This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 datasets in RLDS data format (~10 GB total).
- You can use these datasets to fine-tune OpenVLA or train other methods.
Place your RLDS dataset in the following directory (or use the default clone location):
```
/fs/classhomes/$USER/projects/tmi_openvla/modified_libero_rlds
```

---

## 2. Set Up Directories for Runs and Checkpoints

Use your scratch space for storing large files, runs, and checkpoints. The examples below use the environment variable `$USER` so the instructions work for any user:

```
/scratch0/$USER/openvla_runs
/scratch0/$USER/openvla_adapter_tmp
```

Create these directories if they do not exist:

```bash
# This will automatically use your username
mkdir -p /scratch0/$USER/openvla_runs /scratch0/$USER/openvla_adapter_tmp
```

> **Note:** `$USER` is a standard environment variable that expands to your username on most Linux systems. If it does not work, replace `$USER` with your actual username.

---

## 3. Fine-tune the Model


You can control which prompt variants are used for training by passing the `--prompt_json_type` argument:

- `--prompt_json_type cmin` (default): minimal conversational prompts
- `--prompt_json_type conv`: full conversational prompts
- `--prompt_json_type para`: paraphrased prompts

Example command (adjust `--dataset_name`, `--batch_size`, and `--max_steps` as needed):

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --data_root_dir /fs/classhomes/$USER/projects/tmi_openvla/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --run_root_dir /scratch0/$USER/openvla_runs \
  --adapter_tmp_dir /scratch0/$USER/openvla_adapter_tmp \
  --batch_size 4 \
  --max_steps 1284 \
  --learning_rate 5e-4 \
  --wandb_project "openvla-libero-object" \
  --wandb_entity "your_wandb_entity"
  
```

---

## 4. Run Evaluation

Set the `PYTHONPATH` and run the evaluation script:

```bash

export PYTHONPATH=/fs/classhomes/$USER/projects/tmi_openvla/LIBERO:$PYTHONPATH

python /fs/classhomes/$USER/projects/tmi_openvla/experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /scratch0/$USER/openvla_runs/libero_object_no_noops_cmin_b4_lr0.0005_lora32 \ 
  --task_suite_name libero_object \
  --center_crop True \
  --num_trials_per_task 1 \
  --paraphrase_json /fs/classhomes/$USER/projects/tmi_openvla/experiments/object_conversational_minimal.json
```
- Adjust `--pretrained_checkpoint` and `--task_suite_name` as needed for your experiment.

---

## Notes
- Use absolute paths for all directories and files on Nexus or similar environments.
- Store large files, runs, and checkpoints in your scratch space for best performance.
- For prompt robustness, use conversational or paraphrased prompt JSONs during both training and evaluation.

---
