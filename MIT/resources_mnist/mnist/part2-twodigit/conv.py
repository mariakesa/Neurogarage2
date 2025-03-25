import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 50
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.input_dimension=(42,28)
        self.conv0=nn.Conv2d(1, 16, (5, 5))
        self.relu0=nn.ReLU()
        self.maxpool0=nn.MaxPool2d((2, 2))
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(64, 128, (1, 1))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d((1, 1))
        self.flatten = Flatten()
        self.linear1 = nn.Linear(5760, 256)
        self.dropout = nn.Dropout(0.75)
        self.relu3 = nn.ReLU()
        self.linear10= nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.25)
        self.relu4 = nn.ReLU()
        self.decoder1 = nn.Linear(128, 10)
        self.decoder2 = nn.Linear(128, 10)

    def forward(self, x):
        out=self.conv0(x)
        out=self.relu0(out)
        out=self.maxpool0(out)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out=self.flatten(out)
        out = self.linear1(out)
        out=self.dropout(out)
        out = self.relu3(out)
        out = self.linear10(out)
        out = self.dropout(out)
        self.relu4(out)
        out_first_digit = self.decoder1(out)
        out_second_digit = self.decoder2(out)

        # TODO use model layers to predict the two digits

        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
