import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time

from config import get_system_device
from image_classifier import ImageClassifier


def load_model(clf: ImageClassifier, start_epoch=0, checkpoints_dir="checkpoints"):
    """
    Loads the model and optimizer state if resuming training.
    :param clf: the model to load the state into
    :param start_epoch: if set to n, loads the model from checkpoint_{n-1}.pt
    :param checkpoints_dir:
    :return:
    """
    if not os.path.exists(checkpoints_dir):
        print(f"checkpoints directory not found, creating it, and loading the model randomly initialized")
        os.mkdir(checkpoints_dir)
        return clf, 0
    if os.path.exists(checkpoints_dir):
        # find the last checkpoint that we saved and load it
        for i in range(start_epoch, 0, -1):
            tentative_last_checkpoint_path = os.path.join(os.getcwd(), checkpoints_dir,
                                                          f"{clf.numbers_to_recognize}digit_epoch_{i}.pt")
            if os.path.exists(tentative_last_checkpoint_path):
                checkpoint = torch.load(tentative_last_checkpoint_path)
                clf.load_state_dict(checkpoint['model_state_dict'])
                clf.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"loading the model from : checkpoint_{i - 1}.pt. "
                      f"Eventual training would resume  from epoch {start_epoch}")
                return clf, start_epoch
        print(f"no checkpoints found, loading the model randomly initialized")
        return clf, 0


def train_model(clf: ImageClassifier, datasets, start_epoch=0, epochs=10, checkpoints_dir="checkpoints",
                device=get_system_device(), save_checkpoint_every_n_epochs=5):
    # TODO: parallelize the training loop
    # TODO: add validation loop

    # Create a SummaryWriter instance.
    logs_path = os.path.join(os.getcwd(), "logs", "fit")
    writer = SummaryWriter(log_dir=logs_path)
    # To access TensorBoard, run the following command in terminal while in folder source
    # tensorboard --logdir=logs/fit

    for epoch in range(start_epoch, start_epoch + epochs):
        # Used to calculate the accuracy
        total_predictions = 0
        correct_predictions = 0

        # Time the training loop
        start_time = time.perf_counter_ns()

        for batch in datasets:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = clf(x)
            loss = clf.loss(output, y)

            # Backpropagation
            clf.optimizer.zero_grad()
            loss.backward()
            clf.optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_predictions += y.size(0)
            correct_predictions += (predicted == y).sum().item()

            # Log loss to tensorboard
            writer.add_scalar("Loss/train", loss.item(), epoch)

        # Log accuracy to tensorboard
        writer.add_scalar("Accuracy/train", (output.argmax(1) == y).sum().item() / len(y), epoch)
        print(
            f"Epoch: {epoch} - Loss:  {round(loss.item(), 3)} - Accuracy: {correct_predictions / total_predictions} "
            f"- Time: {round((time.perf_counter_ns() - start_time) / 1e9, 3)}s")

        # save model and optimizer state after save_checkpoint_every_n_epochs epochs
        if epoch % save_checkpoint_every_n_epochs == 0:
            checkpoint_path = os.path.join(os.getcwd(), checkpoints_dir,
                                           f"{clf.numbers_to_recognize}digit_epoch_{epoch}.pt")
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': clf.state_dict(),
                    'optimizer_state_dict': clf.optimizer.state_dict(),
                    'loss': loss,
                },
                checkpoint_path
            )

    # Close the writer instance
    writer.flush()
    writer.close()
