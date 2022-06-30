import torch
import torch.nn.functional as F
import torch.nn as nn
import ConfigSpace as CS
from torch.autograd import Variable

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from warmstarting.checkpoint_gatekeeper import CheckpointGatekeeper


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

if __name__ == '__main__':
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    names = iris['target_names']
    feature_names = iris['feature_names']

    # Scale data to have mean 0 and variance 1
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2)

    lr = CS.hyperparameters.UniformFloatHyperparameter('lr', lower=0.1, upper=1, log=True)
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(lr)

    config = cs.sample_configuration()

    model_full_training = Model(X_train.shape[1])
    optimizer_full_training = torch.optim.Adam(model_full_training.parameters(), lr=0.001)

    model_checkpoint = Model(X_train.shape[1])
    model_checkpoint.load_state_dict(model_full_training.state_dict())
    optimizer_checkpoint = torch.optim.Adam(model_checkpoint.parameters(), lr=0.001)
    optimizer_checkpoint.load_state_dict(optimizer_full_training.state_dict())

    gatekeeper = CheckpointGatekeeper(path=".")

    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 100
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).long()

    loss_list_full = np.zeros((EPOCHS,))
    loss_list_check = np.zeros((EPOCHS,))
    accuracy_list_full = np.zeros((EPOCHS,))
    accuracy_list_check = np.zeros((EPOCHS,))

    for epoch in range(EPOCHS):
        print("Starting epoch: " + str(epoch))
        print("Training full training")
        y_pred = model_full_training(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list_full[epoch] = loss.item()

        # Zero gradients
        optimizer_full_training.zero_grad()
        loss.backward()
        optimizer_full_training.step()

        with torch.no_grad():
            y_pred = model_full_training(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list_full[epoch] = correct.mean()

        print("Training checkpoint training")
        if epoch % 10 == 0:
            print("Saving Checkpoint")
            fidelities = {'epoch': epoch}
            gatekeeper.save_model_state(model_checkpoint, optimizer_checkpoint, config, None, fidelities)
            print("Loading Checkpoint")
            model_checkpoint, optimizer_checkpoint, _, _ = gatekeeper.load_model_state(model_checkpoint, optimizer_checkpoint, config)

        y_pred = model_checkpoint(X_train)
        loss = loss_fn(y_pred, y_train)
        loss_list_check[epoch] = loss.item()

        # Zero gradients
        optimizer_checkpoint.zero_grad()
        loss.backward()
        optimizer_checkpoint.step()

        with torch.no_grad():
            y_pred = model_checkpoint(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list_check[epoch] = correct.mean()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

    ax1.plot(accuracy_list_full, label="full training")
    ax1.plot(accuracy_list_check, label="checkpoint training")
    ax1.set_ylabel("validation accuracy")
    ax1.grid()
    ax2.plot(loss_list_full, label="full training")
    ax2.plot(loss_list_check, label="checkpoint training")
    ax2.set_ylabel("validation loss")
    ax2.set_xlabel("epochs")
    ax2.grid()

    plt.legend()
    plt.show()
