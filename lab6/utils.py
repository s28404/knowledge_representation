import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml
import os
import json



def load_hyperparameters(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_stats(stats, stats_dir="stats/train", file_name="train_stats", seed=None):
    os.makedirs(stats_dir, exist_ok=True)
    path = os.path.join(stats_dir, f"{file_name}_{seed}.json")
    with open(path, "w") as f:
        json.dump(stats, f)


def load_stats(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_loss(stats, file_name=None, seed=None):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(stats["avg_losses"], label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}_{seed}.png")
    plt.close()


def plot_recons(originals, reconstructions, file_name="reconstructions", n=10, seed=None):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 2 rows, n columns
        # i + 1 because subplot indices start at 1
        plt.subplot(2, n, i + 1)
        # from [-1, 1] to [0, 1]
        plt.imshow((originals[i] * 0.5 + 0.5))
        plt.title("Original")
        
        plt.subplot(2, n, i + 1 + n)
        plt.imshow((reconstructions[i] * 0.5 + 0.5))
        plt.title("Reconstructed")
    plt.tight_layout()
    plt.savefig(f"plots/{file_name}_{seed}.png")
    plt.close()


