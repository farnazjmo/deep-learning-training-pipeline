from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    # Logging training and validation metrics.

    # During training:
    # - Log the loss value at each iteration using the tag 'train_loss'
    # - Log average accuracy per epoch using the tag 'train_accuracy'
    
    # During validation:
    # - Log average accuracy per epoch using the tag 'val_accuracy'
    
    # Ensure global_step is used consistently across iterations and epochs
    # (e.g., global_step = 0 at epoch 0, iteration 0)
    """
    # strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        metrics = {"train_acc": [], "val_acc": []}

        # example training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)

            # log train_loss
            logger.add_scalar("train_loss", dummy_train_loss, global_step)

            # save additional metrics to be averaged
            metrics["train_acc"].extend(dummy_train_accuracy.tolist())

            global_step += 1

        # log average train_accuracy
        if metrics["train_acc"]:  # Ensure we have collected values
            avg_train_accuracy = torch.tensor(metrics["train_acc"]).mean()
            logger.add_scalar("train_accuracy", avg_train_accuracy.item(), global_step - 1)

        # example validation loop
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(10)

            # save additional metrics to be averaged
            metrics["val_acc"].extend(dummy_validation_accuracy.tolist())

        # log average val_accuracy
        if metrics["val_acc"]:  # Ensure we have collected values
            avg_val_accuracy = torch.tensor(metrics["val_acc"]).mean()
            logger.add_scalar("val_accuracy", avg_val_accuracy.item(), global_step - 1)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
