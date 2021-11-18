#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 5
#SBATCH --time 72:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/DGPNet/logs/slurm-%j-run.out
#

echo ""
echo "This job was started as: python3 -u /workspace/DGPNet/$1 ${@:2}"
echo ""

singularity exec --bind /workspaces/$USER:/workspace \
	    --bind /data/pascal:/my_data/pascal \
	    --bind /data/coco:/my_data/coco \
	    /workspaces/$USER/DGPNet/singularity/pytorch21_09.sif python3 -u /workspace/DGPNet/$1 ${@:2}
#
#EOF
