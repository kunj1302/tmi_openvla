# LIBERO Evaluation Setup Guide

This guide covers setting up and running OpenVLA LIBERO evaluation on a cluster with GPU compute nodes.

## Prerequisites

- Access to a cluster with GPU compute nodes
- SLURM job scheduler
- ~30GB disk space in home directory
- Python 3.10 available via modules
- Git installed

## Step 1: Clone Repository

Clone your customized OpenVLA repo (which already includes LIBERO as a submodule):

```bash
git clone --recurse-submodules git@github.com:kunj1302/tmi_openvla.git
cd tmi_openvla
```

> If you already cloned without `--recurse-submodules`, run  
> `git submodule update --init --recursive`.

## Step 2: Request Compute Node

Request a GPU compute node with sufficient memory:

```bash
  srun -p class --pty --gres=gpu:rtxa5000:1 --mem=32G --time=04:00:00 /bin/bash
```

Adjust GPU type and memory based on your cluster's available resources.

## Step 3: Setup Python Environment

```bash
# Load Python 3.10 module
module load Python3/3.10.14

# Navigate to repo directory (adjust path as needed)
cd tmi_openvla

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 4: Install Dependencies

### Install PyTorch

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Install OpenVLA

```bash
pip install -e .
```

### Install Flash Attention 2

**Important:** Flash Attention 2 requires CUDA toolkit for compilation.

```bash
# Load CUDA module (adjust version to match your PyTorch CUDA version)
module load cuda/12.1.1

# Install prerequisites
pip install packaging ninja
ninja --version  # Should return exit code 0

# Install Flash Attention 2
# Set TMPDIR to avoid cross-device link errors
export TMPDIR=/tmp
pip install "flash-attn==2.5.5" --no-build-isolation
```

**Note:** If you encounter "No space left on device" errors:
- Clean pip cache: `pip cache purge`
- Set `TMPDIR=/tmp` (which typically has more space)
- Use `--no-cache-dir` flag: `pip install "flash-attn==2.5.5" --no-build-isolation --no-cache-dir`

### Install LIBERO

```bash
# Install LIBERO package
cd LIBERO
pip install -e .
cd ..

# Install LIBERO requirements
pip install -r experiments/robot/libero/libero_requirements.txt

# Downgrade NumPy (required for TensorFlow compatibility)
pip install "numpy<2.0"
```

### Create LIBERO Config

```bash
# Create LIBERO config directory
mkdir -p ~/.libero

# Create config file (adjust paths based on your openvla directory location)
cat > ~/.libero/config.yaml << EOF
benchmark_root: $(pwd)/LIBERO/libero/libero
bddl_files: $(pwd)/LIBERO/libero/libero/bddl_files
init_states: $(pwd)/LIBERO/libero/libero/init_files
datasets: $(pwd)/LIBERO/datasets
assets: $(pwd)/LIBERO/libero/libero/assets
EOF
```

**Note:** Replace `$(pwd)` with your actual openvla directory path if the above doesn't work, or manually edit the config file.

## Step 5: Run LIBERO Evaluation

### Basic Usage

```bash
# Activate environment
cd tmi_openvla  # Navigate to repo
source venv/bin/activate

# Set environment variables
export TMPDIR=/tmp  # Use /tmp for downloads (has more space)
export PYTHONPATH="$(pwd)/LIBERO:$PYTHONPATH"  # Adjust path as needed
```

#### Run LIBERO-Spatial Evaluation

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1
```

#### Run LIBERO-Object Evaluation

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --num_trials_per_task 1
```

#### Run LIBERO-Goal Evaluation

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True \
  --num_trials_per_task 1
```

#### Run LIBERO-10 (Long-Horizon) Evaluation

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --num_trials_per_task 1
```

**Note:** For `libero_90` (90 diverse tasks), there is no fine-tuned checkpoint available. You would need to use the base `openvla/openvla-7b` checkpoint, but note that it does not have LIBERO normalization statistics and may not work correctly.

#### Run LIBERO Evaluation with Paraphrased Prompts

You can test the model's robustness to natural language variations by providing a JSON file with paraphrased task instructions. This is useful for evaluating whether the model can handle synonyms and different sentence structures without changing the core task.

**Using the object_paraphrased.json file:**

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --num_trials_per_task 1 \
  --paraphrase_json experiments/object_paraphrased.json
```

**How it works:**
- The evaluation script will test each task with both the original prompt and all paraphrased variants from the JSON file
- Success rates are tracked separately for original vs. paraphrased prompts
- Results are saved to a detailed JSON file: `./experiments/logs/EVAL-{task_suite}-{model}-{timestamp}_paraphrase_results.json`
- The log file includes a summary comparing original vs. paraphrased prompt performance

**JSON Format:**
The paraphrase JSON file should have the following structure:
```json
{
  "pick up the alphabet soup and place it in the basket": [
    "grab the alphabet soup and put it in the basket",
    "move the alphabet soup into the basket",
    ...
  ],
  ...
}
```

Where each key is the original task instruction and the value is a list of paraphrased versions.

**Note:** The task description in the JSON file must exactly match the task description from the LIBERO benchmark. If a task doesn't have a matching key in the JSON file, a warning will be printed and only the original prompt will be tested.

See `PARAPHRASE_GENERATION.md` for details on how the paraphrases were generated.

#### Evaluation Results

**LIBERO-Object with Paraphrased Prompts (1 trial per prompt variant):**

- **Original Prompts:** 7/10 (70.00%)
- **Paraphrased Prompts:** 27/50 (54.00%)

These results show that the model performs better with the original task descriptions compared to paraphrased variants, indicating some sensitivity to natural language variations. The paraphrased prompts achieve a 54% success rate across 50 trials (10 tasks × 5 paraphrases each), while the original prompts achieve 70% success rate across 10 trials.

#### Run LIBERO Evaluation with Conversational Prompts

You can test the model's robustness to conversational language by providing a JSON file with conversational task instructions. This is useful for evaluating whether the model can handle natural conversational elements like greetings, politeness phrases, and additional context without changing the core task.

Using the `object_conversational.json` file:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --num_trials_per_task 1 \
  --paraphrase_json experiments/object_conversational.json
```

**How it works:**

- The evaluation script will test each task with both the original prompt and all conversational variants from the JSON file
- Success rates are tracked separately for original vs. conversational prompts
- Results are saved to a detailed JSON file: `./experiments/logs/EVAL-{task_suite}-{model}-{timestamp}_paraphrase_results.json`
- The log file includes a summary comparing original vs. conversational prompt performance

**JSON Format:** The conversational JSON file should have the following structure:

```json
{
  "pick up the alphabet soup and place it in the basket": [
    "Hello! I'd like you to pick up the alphabet soup and place it in the basket, if you don't mind.",
    "Could you please grab the alphabet soup? I need it placed in the basket when you get a chance.",
    "Just so you know, we need to pick up the alphabet soup and move it to the basket.",
    "Hey there! The alphabet soup needs to go in the basket. Can you handle that?",
    "I was wondering if you could pick up that alphabet soup and put it in the basket for me."
  ],
  ...
}
```

Where each key is the original task instruction and the value is a list of conversational variants with additional phrases like greetings, politeness markers, and contextual commentary.

**Note:** The task description in the JSON file must exactly match the task description from the LIBERO benchmark. If a task doesn't have a matching key in the JSON file, a warning will be printed and only the original prompt will be tested.

See `CONVERSATIONAL_GENERATION.md` for details on how the conversational variants were generated.

#### Evaluation Results Comparison

**LIBERO-Object with Original, Paraphrased, and Conversational Prompts (1 trial per prompt variant):**

- **Original Prompts:** 6/10 (60.00%)
- **Paraphrased Prompts:** 27/50 (54.00%)  
- **Conversational Prompts:** 8/50 (16.00%)

These results demonstrate the model's varying sensitivity to different types of natural language variations. While paraphrased prompts (with synonym substitutions) achieve 54% success rate comparable to the 60% with original prompts, conversational prompts (with politeness phrases, greetings, and additional context) show a dramatic drop to 16% success rate. This indicates that the model struggles significantly with the additional linguistic elements present in natural conversational instructions, even when the core task description remains unchanged.

### Available Task Suites

- `libero_spatial` - 10 spatial reasoning tasks
- `libero_object` - 10 object manipulation tasks  
- `libero_goal` - 10 goal-oriented tasks
- `libero_10` - 10 long-horizon tasks
- `libero_90` - 90 diverse tasks

### Available Checkpoints

- `openvla/openvla-7b-finetuned-libero-spatial` - For `libero_spatial`
- `openvla/openvla-7b-finetuned-libero-object` - For `libero_object`
- `openvla/openvla-7b-finetuned-libero-goal` - For `libero_goal`
- `openvla/openvla-7b-finetuned-libero-10` - For `libero_10`

**Important:** Use the matching checkpoint for your task suite. The base `openvla/openvla-7b` checkpoint does not have LIBERO normalization statistics.

### Evaluation Parameters

- `--num_trials_per_task`: Number of rollouts per task (default: 50, use 1 for quick testing)
- `--center_crop`: Set to `True` if model was fine-tuned with augmentations (default: `True`)
- `--seed`: Random seed for reproducibility (default: 7)
- `--paraphrase_json`: Optional path to JSON file with prompt variants (paraphrased or conversational) to test (default: `None`)

## Step 6: Results

Results are saved to:
- **Log file**: `./experiments/logs/EVAL-{task_suite}-{model}-{timestamp}.txt`
- **Videos**: `./rollouts/{date}/episode_{N}_success_{True/False}_task_{description}.mp4`
- **Prompt variants results** (if `--paraphrase_json` is used): `./experiments/logs/EVAL-{task_suite}-{model}-{timestamp}_paraphrase_results.json`

## Troubleshooting

### "No space left on device" errors

1. Check disk usage: `df -h /fs/classhomes/<username>`
2. Clean pip cache: `pip cache purge`
3. Remove unused model caches: `rm -rf ~/.cache/huggingface/hub/models--openvla--openvla-7b`
4. Set `TMPDIR=/tmp` and `HF_HOME=/tmp/huggingface` before running (as shown above) to use /tmp for downloads and model cache
5. If home directory is still full, you can move the existing cache: `mv ~/.cache/huggingface /tmp/huggingface` and then set `HF_HOME=/tmp/huggingface`

### Flash Attention 2 installation fails

- Ensure CUDA module is loaded: `module load cuda/12.1.1`
- Verify `nvcc` is available: `which nvcc`
- Set `TMPDIR=/tmp` before installation
- Use `--no-cache-dir` flag if space is limited

### "ModuleNotFoundError: No module named 'libero'"

- Ensure `PYTHONPATH` includes LIBERO: `export PYTHONPATH="$(pwd)/LIBERO:$PYTHONPATH"` (adjust path as needed)
- Verify LIBERO is installed: `pip list | grep libero`

### NumPy compatibility errors

- Ensure NumPy < 2.0: `pip install "numpy<2.0"`
- This is required for TensorFlow compatibility

## Quick Reference

**How it works:**
- The evaluation script will test each task with both the original prompt and all paraphrased variants from the JSON file
- Success rates are tracked separately for original vs. paraphrased prompts
- Results are saved to a detailed JSON file: `./experiments/logs/EVAL-{task_suite}-{model}-{timestamp}_paraphrase_results.json`
- The log file includes a summary comparing original vs. paraphrased prompt performance

**JSON Format:**
The paraphrase JSON file should have the following structure:
```json
{
  "pick up the alphabet soup and place it in the basket": [
    "grab the alphabet soup and put it in the basket",
    "move the alphabet soup into the basket",
    ...
  ],
  ...
}
```

Where each key is the original task instruction and the value is a list of paraphrased versions.

**Note:** The task description in the JSON file must exactly match the task description from the LIBERO benchmark. If a task doesn't have a matching key in the JSON file, a warning will be printed and only the original prompt will be tested.

See `PARAPHRASE_GENERATION.md` for details on how the paraphrases were generated.

#### Evaluation Results

**LIBERO-Object with Paraphrased Prompts (1 trial per prompt variant):**

- **Original Prompts:** 7/10 (70.00%)
- **Paraphrased Prompts:** 27/50 (54.00%)

These results show that the model performs better with the original task descriptions compared to paraphrased variants, indicating some sensitivity to natural language variations. The paraphrased prompts achieve a 54% success rate across 50 trials (10 tasks × 5 paraphrases each), while the original prompts achieve 70% success rate across 10 trials.
```bash
# Complete setup (after getting compute node)

# Clone (if not already)
git clone --recurse-submodules git@github.com:kunj1302/tmi_openvla.git
cd tmi_openvla

# Setup environment
module load Python3/3.10.14
module load cuda/12.1.1
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
cd LIBERO && pip install -e . && cd ..
pip install -r experiments/robot/libero/libero_requirements.txt
pip install "numpy<2.0"
export TMPDIR=/tmp
export HF_HOME=/tmp/huggingface  # Use /tmp for Hugging Face cache (avoids filling home directory)
export PYTHONPATH="$(pwd)/LIBERO:$PYTHONPATH"
```

### Run Evaluations

**LIBERO-Spatial:**
```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1
```

**LIBERO-Object:**
```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --num_trials_per_task 1
```

**LIBERO-Object with Paraphrased Prompts:**
```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --num_trials_per_task 1 \
  --paraphrase_json experiments/object_paraphrased.json
```

**LIBERO-Object with Conversational Prompts:**
```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True \
  --num_trials_per_task 1 \
  --paraphrase_json experiments/object_conversational.json
```

**LIBERO-Goal:**
```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True \
  --num_trials_per_task 1
```

**LIBERO-10 (Long-Horizon):**
```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True \
  --num_trials_per_task 1
```

