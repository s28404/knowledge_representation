import matplotlib.pyplot as plt
import yaml
import os
import json


def load_hyperparameters(file_path):
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        exit(1)

    return config


def convert_to_serializable(obj):
    # Recursively convert dictionary values on float32 to Python float because json.dump cannot handle float32
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [float(item) if hasattr(item, "item") else item for item in obj] 
    else:
        return obj


def save_stats(stats, stats_dir="stats/train", file_name="train_stats.json"):
    os.makedirs(stats_dir, exist_ok=True)
    path = os.path.join(stats_dir, file_name)
    # Convert stats to JSON-serializable format
    serializable_stats = convert_to_serializable(stats)
    with open(path, "w") as f:
        json.dump(serializable_stats, f, indent=2)


def load_stats(path):
    with open(path, "r") as f:
        stats = json.load(f)
    return stats


def plot(stats, training=True, file_name=None):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    if training:
        plt.plot(stats["avg_losses"], label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{file_name}")
    else:
        plt.plot(stats["avg_losses"], label="Testing Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Testing Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{file_name}")
    
    plt.close()
