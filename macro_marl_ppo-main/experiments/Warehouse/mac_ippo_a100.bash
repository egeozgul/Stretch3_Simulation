#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100@80g
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --output=discovery/results/%j/myjob.%j.out

module load anaconda3/2022.05 cuda/12.3

# Activate the specified Anaconda environment
source activate /scratch/lin.wo/macro_marl

# Log job start time
echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S')"

# Execute the script
echo "Executing mac_ippov2.sh"
./mac_ippov2.sh

# Log job end time
echo "Job ended at: $(date '+%Y-%m-%d %H:%M:%S')"
