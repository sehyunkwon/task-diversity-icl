import json
import os
import random 
from itertools import combinations
import argparse


# ────────── argparse setup ──────────
def parse_args():
    parser = argparse.ArgumentParser(description='Create combined json files.')
    parser.add_argument('--n',   type=int,   required=True,
                        help='Number of JSON files to combine.')
    parser.add_argument('--gpu', nargs='+',  type=int, required=True,
                        help='List of GPUs to use.')
    return parser.parse_args()


# ────────── Fixed values at the top ──────────
fixed_json_data = {
    "family": "san",
    "model_name": "gpt2",
    "train_steps": 1_000_000,
    "n_dims": 10,
    "n_embd": 256,
    "n_layer": 12,
    "n_head": 8,
    "learning_rate": 1e-4,
    "B": 64,                      # The total batch size is always 64
    "gpu": [],                    # To be replaced by argparse
    "curriculum_points_start": 120,
    "curriculum_points_end": 120,
}

# ────────── 🔹 New function: Adjust B per task ──────────
def adjust_batch_sizes(task_list, total_batch=64, seed=None):
    """
    Set each task's B (batch size) to '⌊64 / n⌋',
    and if 64 is not a multiple of n,
    randomly pick tasks to add +1 so the sum becomes exactly 64.
    """
    if seed is not None:
        random.seed(seed)

    n = len(task_list)
    base = total_batch // n
    remainder = total_batch - base * n

    # Assign base value
    for task in task_list:
        task['batch_size'] = base

    # Randomly add +1 to some tasks if there's a remainder
    if remainder:
        for idx in random.sample(range(n), remainder):
            task_list[idx]['batch_size'] += 1

    # Debug / verification
    # assert sum(t['batch_size'] for t in task_list) == total_batch
    return task_list


# ────────── Create combinations of n ──────────
def create_combined_json_files(n, gpu_list, input_path, output_path):
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
    os.makedirs(output_path, exist_ok=True)

    for selected_files in combinations(json_files, n):
        combined_task_list, task_names = [], []

        # Merge task_lists from selected files
        for file_name in selected_files:
            with open(os.path.join(input_path, file_name), 'r') as f:
                data = json.load(f)
                combined_task_list.extend(data['task_list'])
                task_names.append(data['task_list'][0]['exp_name'])

        # 🔹 Automatically adjust batch sizes
        combined_task_list = adjust_batch_sizes(combined_task_list, total_batch=64)

        task_name_str = '-'.join(task_names)
        new_json_name = f"mix{n}-{task_name_str}.json"

        # Create new json data
        new_json_data = fixed_json_data.copy()
        new_json_data.update({
            'name': f"mix{n}-{task_name_str}",
            'task_list': combined_task_list,
            'gpu': gpu_list,        # From argparse
        })

        # Save
        with open(os.path.join(output_path, new_json_name), 'w') as out_f:
            json.dump(new_json_data, out_f, indent=4)

    print(f"{len(list(combinations(json_files, n)))} combined json files have been created.")


# ────────── main ──────────
if __name__ == '__main__':
    args = parse_args()

    input_path  = '/extdata2/sehyun/task_diversity/task_configs/references'
    output_path = '/extdata2/sehyun/task_diversity/task_configs/num=6'

    create_combined_json_files(args.n, args.gpu, input_path, output_path)
