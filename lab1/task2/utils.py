import matplotlib

matplotlib.use("Agg")  # non-interactive backend to avoid Qt/GUI issues

import matplotlib.pyplot as plt
import yaml
import os
import json
import numpy as np
import tensorflow as tf
import getpass
import socket
from datetime import datetime


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
    # recursively convert dictionary values on float32 to Python float because json.dump cannot handle float32
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [float(item) if hasattr(item, "item") else item for item in obj]
    else:
        return obj


def save_stats(stats, stats_dir="stats/train", file_name="train_stats", seed=None):
    os.makedirs(stats_dir, exist_ok=True)
    path = os.path.join(stats_dir, f"{file_name}_seed_{seed}.json")
    # Convert stats to JSON-serializable format
    serializable_stats = convert_to_serializable(stats)
    with open(path, "w") as f:
        json.dump(serializable_stats, f, indent=2)


def load_stats(path):
    with open(path, "r") as f:
        stats = json.load(f)
    return stats


def plot_loss(stats, file_name=None, seed=None):
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 5))

    plt.plot(stats["avg_losses"], label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}_seed_{seed}.png")

def plot_recons(originals, reconstructions, file_name="reconstructions", seed=None, n=10):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow((originals[i] * 0.5 + 0.5))  # unnormalize from [-1, 1] to [0, 1]
        plt.title("Original")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow((reconstructions[i] * 0.5 + 0.5))  # unnormalize from [-1, 1] to [0, 1]
        plt.title("Reconstructed")
 
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}_seed_{seed}.png")
    plt.close()

def save_metadata(model, config, seed, metadata_dir="metadata"):
    os.makedirs(metadata_dir, exist_ok=True)

    trainable_params = int(
        np.sum([np.prod(v.shape) for v in model.trainable_variables])
    )
    non_trainable_params = int(
        np.sum([np.prod(v.shape) for v in model.non_trainable_variables])
    )
    total_params = trainable_params + non_trainable_params

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "framework": "TensorFlow",
        "tf_version": tf.__version__,
        "device": tf.config.list_physical_devices("GPU")[0].name
        if tf.config.list_physical_devices("GPU")
        else "CPU",
        "seed": seed,
        "model_parameters": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
        },
        "config": config,
    }

    path = os.path.join(metadata_dir, f"metadata_seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Metadata saved to: {path}\n")

    return metadata
