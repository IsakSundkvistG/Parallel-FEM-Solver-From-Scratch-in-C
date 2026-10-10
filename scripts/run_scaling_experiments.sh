#!/usr/bin/env bash

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
executable="$script_directory/parallel/exp_fem_parallel"
mesh_directory="$script_directory/meshes"
output_file="$script_directory/parallel_scaling_results.csv"
repetitions=5

export OMP_DYNAMIC=FALSE
unset OMP_PROC_BIND
unset OMP_PLACES

if [[ ! -s "$output_file" ]]
then
    echo "experiment_type,threads,nodes,repetition,CG_runtime,tot_runtime,success" \
        > "$output_file"
fi

run_experiment()
{
    experiment_type=$1
    threads=$2
    mesh_file=$3

    export OMP_NUM_THREADS=$threads

    for ((repetition = 1; repetition <= repetitions; repetition++))
    do
        # Skip runs already recorded in the CSV.
        if awk -F',' \
            -v type="$experiment_type" \
            -v p="$threads" \
            -v rep="$repetition" \
            'NR > 1 && $1 == type && $2 == p && $4 == rep { found = 1 }
             END { exit !found }' \
            "$output_file"
        then
            continue
        fi

        if result=$("$executable" "$mesh_file" 2>/dev/null)
        then
            IFS=',' read -r nodes CG_runtime tot_runtime success <<< "$result"

            echo "$experiment_type,$threads,$nodes,$repetition,$CG_runtime,$tot_runtime,$success" \
                >> "$output_file"
        else
            echo "$experiment_type,$threads,,$repetition,,,0" \
                >> "$output_file"
        fi
    done
}

# Strong scaling.
run_experiment "strong" 1  "$mesh_directory/disk_1600k.msh"
run_experiment "strong" 2  "$mesh_directory/disk_1600k.msh"
run_experiment "strong" 4  "$mesh_directory/disk_1600k.msh"
run_experiment "strong" 8  "$mesh_directory/disk_1600k.msh"
run_experiment "strong" 16 "$mesh_directory/disk_1600k.msh"
run_experiment "strong" 32 "$mesh_directory/disk_1600k.msh"

# Weak scaling: approximately 100,000 nodes per thread.
run_experiment "weak" 1  "$mesh_directory/disk_100k.msh"
run_experiment "weak" 2  "$mesh_directory/disk_200k.msh"
run_experiment "weak" 4  "$mesh_directory/disk_400k.msh"
run_experiment "weak" 8  "$mesh_directory/disk_800k.msh"
run_experiment "weak" 16 "$mesh_directory/disk_1600k.msh"