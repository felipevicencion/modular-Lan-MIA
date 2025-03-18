import os
import json
import csv

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def summarize_sumstats_to_csv(directory="../attack_bench/"):
    summary_data = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".sumstats.json"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    parts = filename.rsplit("__", 2)
                    if len(parts) == 3:
                        dataset, model, attack = parts[0], parts[1], parts[2].replace(".sumstats.json", "")
                    else:
                        dataset, model, attack = "unknown", "unknown", "unknown"
                    metrics = flatten_dict(data) if isinstance(data, dict) else {}
                    summary_data.append({"dataset": dataset, "model": model, "attack": attack, **metrics})
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    output_csv = os.path.join(directory, "summary_metrics.csv")
    if summary_data:
        keys = sorted(set().union(*[d.keys() for d in summary_data]))
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"Summary saved to {output_csv}")
    else:
        print("No valid .sumstats.json files found.")

summarize_sumstats_to_csv()
