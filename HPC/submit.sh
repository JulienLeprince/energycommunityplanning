#!/bin/bash


email_address=amosc@dtu.dk

declare -a jobs=("stochastic_distributed_full_hourly" "stochastic_distributed_poc_hourly" "stochastic_centralized_poc_hourly" "sensitivity_full_hourly" "sensitivity_test_hourly" "stochastic_distributed_full_15mins" "stochastic_distributed_poc_15mins" "stochastic_centralized_poc_15mins" "sensitivity_full_15mins" "sensitivity_test_15mins")

for job in "${jobs[@]}"
  do
    bsub -J $job_name -q compute -o ${job_name}.out -e ${job_name}.err -n 2 -R "span[hosts=1]" -R "rusage[mem=120GB]" -M 121GB -u ${email_address} -W 72:00 -R "select[model == XeonE5_2650v4]" -B -N HPC/script_${job}.sub
