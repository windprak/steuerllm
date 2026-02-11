#!/bin/bash -l
#SBATCH --job-name=steuerllmtrainbase
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --partition=h100
#SBATCH -o /path/to/your/workspace/out-%x-%j-on-%N.out
#SBATCH --time=24:00:00
#SBATCH --export=ALL

# === General Setup ===
unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=32
export https_proxy="http://your-proxy-server:80"
echo "Your job is running on" $(hostname)
module load cuda/12.6.2
export TERM=xterm

# === Caches and Directories ===
export HF_DATASETS_CACHE="/path/to/your/cache"
export TRITON_CACHE_DIR=$TMPDIR
export CUTLASS_PATH="/path/to/your/cutlass"
export HF_HOME="/path/to/your/.huggingface"

# === Environment for torchrun with Infiniband (FSDP2 COMPATIBLE) ===
export TORCHELASTIC_TIMEOUT=3600
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1 
export NCCL_ENABLE_RETRY=1
export NCCL_RETRY_COUNT=20
export NCCL_RETRY_TIMEOUT=3600
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1

# FIXED NCCL SETTINGS FOR FSDP2
# export NCCL_PROTO=simple           # REMOVED - let NCCL auto-detect
# export NCCL_TREE_THRESHOLD=0       # REMOVED - let NCCL choose algorithm

export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# FSDP2 SPECIFIC ADDITIONS
export TORCH_NCCL_USE_COMM_NONBLOCKING=1

# === WANDB / HF setup ===
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info
export WANDB_DATA_DIR=/path/to/your/wandb_data

# === Torchrun Setup ===
GPUS_PER_NODE=4
NNODES=$SLURM_JOB_NUM_NODES
RANK=$SLURM_PROCID
export MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29400

# === Paths ===
AXOLOTL_CFG=/path/to/your/axo_steuerllm_base.yaml
CONTAINER=/path/to/your/axolotllatest.sif
WORKDIR=/path/to/your/wandb_data
LOG_PATH=/path/to/your/wandb_data

echo "===== Environment & Torchrun Setup ====="
echo "Job Name         : $SLURM_JOB_NAME"
echo "SLURM Job ID     : $SLURM_JOB_ID"
echo "Node List        : $SLURM_NODELIST"
echo "Num Nodes        : $SLURM_JOB_NUM_NODES"
echo "Node Rank        : $RANK"
echo "GPUs Per Node    : $GPUS_PER_NODE"
echo "Master Address   : $MASTER_ADDR"
echo "Master Port      : $MASTER_PORT"
echo "Torchrun Args    :"
echo "  --nproc_per_node=$GPUS_PER_NODE"
echo "  --nnodes=$NNODES"
echo "  --node_rank=$RANK"
echo "  --rdzv_id=$SLURM_JOB_ID"
echo "  --rdzv_backend=c10d"
echo "  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
echo "Axolotl Config   : $AXOLOTL_CFG"
echo "Log Path         : $LOG_PATH"
echo "========================================="

# === Run with Apptainer ===
srun --label bash -c 'echo "[Rank=$SLURM_PROCID on $(hostname)]: RANK=$SLURM_PROCID; MASTER_ADDR=$MASTER_ADDR; MASTER_PORT=$MASTER_PORT"'

srun --wait=60 --kill-on-bad-exit=1 apptainer exec \
  --nv \
  --bind /hnvme,$TMPDIR \
  $CONTAINER \
  bash -c "
    cd $WORKDIR && \
    torchrun \
      --nproc_per_node=$GPUS_PER_NODE \
      --nnodes=$NNODES \
      --node_rank=$RANK \
      --rdzv_id=$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      -m axolotl.cli.train $AXOLOTL_CFG \
      2>&1 | tee -a $LOG_PATH
  "

echo "END TIME: $(date)"
