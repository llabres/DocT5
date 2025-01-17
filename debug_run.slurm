#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --account=ehpc44
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=debug_runs/debug%j.out
#SBATCH --error=debug_runs/debug%j.err
export SLURM_CPU_BIND=none

module load cuda/12.3
source /gpfs/projects/ehpc44/hpc_env/bin/activate

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

export GPUS_PER_NODE=4


echo Node IP: $head_node_ip
echo Num Nodes: $SLURM_NNODES
echo Num GPUs: $((SLURM_NNODES * GPUS_PER_NODE))

export LOGLEVEL=INFO

python_script=$1
shift

export WANDB_MODE=offline 

srun accelerate launch --multi-gpu --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) --num_machines $SLURM_NNODES --machine_rank $SLURM_NODEID --rdzv_backend c10d --main_process_ip $head_node_ip --main_process_port 29500 $python_script $@