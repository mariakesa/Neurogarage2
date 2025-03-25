"""Training utilities."""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class Flatten(nn.Module):
    """A custom layer that views an input as 1D."""
    
    def forward(self, input):
        return input.view(input.size(0), -1)


def batchify_data(x_data, y_data, batch_size):
    """Takes a set of data points and labels and groups them into batches."""
    # Only take batch_size chunks (i.e. drop the remainder)
    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': torch.tensor(x_data[i:i + batch_size],
                              dtype=torch.float32),
            'y': torch.tensor([y_data[0][i:i + batch_size],
                               y_data[1][i:i + batch_size]],
                               dtype=torch.int64)
        })
    return batches


def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.cpu().numpy(), y.cpu().numpy()))

def train_model(train_data, dev_data, model, lr=0.001, momentum=0.9, nesterov=False, n_epochs=50):
    """Train a model for N epochs given data and hyper-params."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 🔧 Add scheduler (e.g., StepLR that decays every 10 epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # 🔧 Pass scheduler to run_epoch during training
        loss, acc = run_epoch(train_data, model.train(), optimizer, scheduler)
        print('Train | loss1: {:.6f}  accuracy1: {:.6f} | loss2: {:.6f}  accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

        # 🔧 For validation, pass scheduler=None
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer, scheduler=None)
        print('Valid | loss1: {:.6f}  accuracy1: {:.6f} | loss2: {:.6f}  accuracy2: {:.6f}'.format(val_loss[0], val_acc[0], val_loss[1], val_acc[1]))

        # 🔧 Optional: print current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")

        # Save model
        torch.save(model, 'mnist_model_fully_connected.pt')



def run_epoch(data, model, optimizer, scheduler=None):
    """Train or evaluate model for one pass through data. Return loss and accuracy."""
    losses_first_label = []
    losses_second_label = []
    batch_accuracies_first = []
    batch_accuracies_second = []

    is_training = model.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in tqdm(data):
        x, y = batch['x'].to(device), batch['y'].to(device)

        out1, out2 = model(x)

        predictions_first_label = torch.argmax(out1, dim=1)
        predictions_second_label = torch.argmax(out2, dim=1)
        batch_accuracies_first.append(compute_accuracy(predictions_first_label, y[0]))
        batch_accuracies_second.append(compute_accuracy(predictions_second_label, y[1]))

        loss1 = F.cross_entropy(out1, y[0])
        loss2 = F.cross_entropy(out2, y[1])
        losses_first_label.append(loss1.item())
        losses_second_label.append(loss2.item())

        if is_training:
            optimizer.zero_grad()
            joint_loss = 0.5 * (loss1 + loss2)
            joint_loss.backward()
            optimizer.step()

    # 🔧 Step scheduler once per epoch, not per batch
    if is_training and scheduler is not None:
        scheduler.step(joint_loss)

    avg_loss = np.mean(losses_first_label), np.mean(losses_second_label)
    avg_accuracy = np.mean(batch_accuracies_first), np.mean(batch_accuracies_second)
    return avg_loss, avg_accuracy

