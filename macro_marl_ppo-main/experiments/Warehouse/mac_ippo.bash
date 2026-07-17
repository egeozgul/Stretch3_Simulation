#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=discovery/results/%j/myjob.%j.out

module load anaconda3/2022.05

source activate /scratch/lin.wo/macro_marl

# Improved date format for clarity and added message for start and end of the job
echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S')"

# Corrected echo statement for script execution (removed incorrect markdown link)
echo "Executing mac_ippo.sh"
./mac_ippo.sh

echo "Job ended at: $(date '+%Y-%m-%d %H:%M:%S')"
