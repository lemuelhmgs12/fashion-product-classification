"""
Plotting helpers shared across notebooks.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(history, title, save_path=None):
    """Plot training vs validation loss & accuracy, epoch by epoch, and mark the best epoch."""

    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    # best epoch = highest validation accuracy (the one ModelCheckpoint saved)
    best_epoch = np.argmax(history_dict["val_accuracy"]) + 1     # +1 because epochs start at 1
    best_val_acc = max(history_dict["val_accuracy"])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # loss
    ax[0].plot(epochs, history_dict["loss"], label="Training Loss")
    ax[0].plot(epochs, history_dict["val_loss"], label="Validation Loss")
    ax[0].set_title(f"{title} - Loss")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # accuracy
    ax[1].plot(epochs, history_dict["accuracy"], label="Training Accuracy")
    ax[1].plot(epochs, history_dict["val_accuracy"], label="Validation Accuracy")
    ax[1].axvline(best_epoch, color="grey", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax[1].set_title(f"{title} - Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")