#!/bin/bash
#SBATCH --job-name="parameter_sweep"
#SBATCH --output="output_%j.log"
#SBATCH --error="error_%j.log"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=rome
#SBATCH --mem=16GB

# load Python module
module load 2022
module load Python/3.9.6-GCCcore-11.2.0
python3 -m pip install --user pandas

# set variables for parameter sweep
prob_move_values=(0.5 0.7 0.9)
transaction_probs=(0.6 0.7 0.8)
inequality_params=(0.01 0.05 0.1)
tax_rates=(0.2 0.4 0.6)
delta_m_values=(1 5 10)
timesteps=(200 500 1000)

# fixed grid size
height=20
width=20

# initialize run number
run_num=1

# loop through parameter combinations
for prob_move in "${prob_move_values[@]}"; do
    for transaction_prob in "${transaction_probs[@]}"; do
        for inequality_param in "${inequality_params[@]}"; do
            for tax_rate in "${tax_rates[@]}"; do
                for delta_m in "${delta_m_values[@]}"; do
                    for timestep in "${timesteps[@]}"; do
                        echo "Running with: P_m=$prob_move, P_t=$transaction_prob, P_i=$inequality_param, Tax=$tax_rate, Delta_m=$delta_m, Steps=$timestep"
                        
                        # run Python script
                        python3 animate_CA_simulation.py \
                            --run_num $run_num \
                            --prob_move $prob_move \
                            --transaction_prob $transaction_prob \
                            --inequality_param $inequality_param \
                            --tax_rate $tax_rate \
                            --delta_m $delta_m \
                            --steps $timestep \
                            --height $height \
                            --width $width
                        
                        # add  run number
                        run_num=$((run_num + 1))
                    done
                done
            done
        done
    done
done
