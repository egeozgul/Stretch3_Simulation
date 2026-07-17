#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=discovery/results/%j/myjob.%j.out

module load anaconda3/2022.05

source activate /scratch/lin.wo

echo "Job started at: $(date '+%Y-%m-%d %H:%M:%S')"

# Define hyperparameter arrays
a_learning_rates=(0.0005 0.0004 0.0003)
c_learning_rates=(0.0005 0.0004 0.0003)
ppo_clip_values=(0.1 0.2 0.05)
ppo_epochs=(2 4 6)

# Loop through hyperparameter combinations
for a_lr in "${a_learning_rates[@]}"; do
    for c_lr in "${c_learning_rates[@]}"; do
        for epochs in "${ppo_epochs[@]}"; do
            for freq in "${train_freqs[@]}"; do
                for ((i=0; i<1; i++))  # Run each combination 10 times
                do
                    echo "Starting job with a_lr=${a_lr}, c_lr=${c_lr}, ppo_epochs=${epochs}, train_freq=${freq}, run=${i}"
                    pg_based_main.py --save_dir="Mac_MAPPO_BoxPushing_10x10" \
                        --alg='MacMAPPO' \
                        --env_id='BP-MA-v0' \
                        --n_agent=2 \
                        --device='cuda' \
                        --env_terminate_step=100 \
                        --big_box_reward=300 \
                        --ppo_clip_value=$ppo_clip_values \
                        --tracking \
                        --ppo_epochs=$epochs \
                        --a_lr=$a_lr \
                        --c_lr=$c_lr \
                        --train_freq=16 \
                        --n_env=8 \
                        --c_target_update_freq=32 \
                        --n_step_TD=3 \
                        --grad_clip_norm=0 \
                        --eps_start=0.999 \
                        --eps_end=0.01 \
                        --eps_stable_at=8_000 \
                        --total_epi=40_000 \
                        --grid_dim 12 12 \
                        --gamma=0.99 \
                        --eval_policy \
                        --sample_epi \
                        --run_id=$i &
                done
            done
        done
    done
done

wait

echo "All jobs submitted. Job ended at: $(date '+%Y-%m-%d %H:%M:%S')"