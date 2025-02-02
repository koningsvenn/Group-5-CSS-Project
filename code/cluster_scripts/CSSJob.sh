#!/bin/bash
#SBATCH --job-name="parameter_sweep"
#SBATCH --output="output_%j.log"
#SBATCH --error="error_%j.log"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00  # Increased time limit for longer runs
#SBATCH --partition=rome
#SBATCH --mem=16GB

# Load Python module
module load 2022
module load Python/3.9.6-GCCcore-11.2.0
python3 -m pip install --user pandas

# Set variables for parameter sweep
prob_move_values=(0.7)  # Single value
transaction_probs=(0.8)  # Single value
inequality_params=(0 0.01 0.05 0.1)  # 4 values
tax_rates=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)  # 11 values
delta_m_values=(1)  # Single value
timesteps=(10000)  # Single value
charity_contributions=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)  # 11 values
grid_sizes=(10x10 20x20 30x30 40x40)  # 4 values

# Initialize run number
run_num=1

# Loop through parameter combinations
for grid_size in "${grid_sizes[@]}"; do
    height=${grid_size%x*}  # Extract height from grid size
    width=${grid_size#*x}  # Extract width from grid size
    for inequality_param in "${inequality_params[@]}"; do
        for tax_rate in "${tax_rates[@]}"; do
            for charity_contribution in "${charity_contributions[@]}"; do
                echo "Running with: Grid=$grid_size, P_m=$prob_move_values, P_t=$transaction_probs, P_i=$inequality_param, Tax=$tax_rate, Delta_m=$delta_m_values, Steps=$timesteps, Charity=$charity_contribution"

                # Run Python script
                python3 animate_CA_simulation.py \
                    --run_num $run_num \
                    --prob_move ${prob_move_values[0]} \
                    --transaction_prob ${transaction_probs[0]} \
                    --inequality_param $inequality_param \
                    --tax_rate $tax_rate \
                    --delta_m ${delta_m_values[0]} \
                    --steps ${timesteps[0]} \
                    --height $height \
                    --width $width \
                    --charity_contribution $charity_contribution

                # Increment run number
                run_num=$((run_num + 1))
            done
        done
    done
done
