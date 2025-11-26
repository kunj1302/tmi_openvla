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
srun --pty --gres=gpu:rtxa5000:1 --mem=32G --time=04:00:00 /bin/bash
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

# Run evaluation
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1
```

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

## Step 6: Results

Results are saved to:
- **Log file**: `./experiments/logs/EVAL-{task_suite}-{model}-{timestamp}.txt`
- **Videos**: `./rollouts/{date}/episode_{N}_success_{True/False}_task_{description}.mp4`

## Troubleshooting

### "No space left on device" errors

1. Check disk usage: `df -h /fs/classhomes/<username>`
2. Clean pip cache: `pip cache purge`
3. Remove unused model caches: `rm -rf ~/.cache/huggingface/hub/models--openvla--openvla-7b`
4. Set `TMPDIR=/tmp` before running (as shown above)

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

```bash
# Complete setup and run (after getting compute node)

# Clone (if not already)
git clone --recurse-submodules git@github.com:kunj1302/tmi_openvla.git
cd tmi_openvla

# Setup and run
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
export PYTHONPATH="$(pwd)/LIBERO:$PYTHONPATH"

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1
```

