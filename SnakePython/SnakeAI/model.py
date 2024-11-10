"""
This module defines a linear Q-network using PyTorch for reinforcement learning tasks. 
It includes two primary classes: Linear_QNet, which represents the architecture of the neural network, 
and QTrainer, which manages the training process. The Linear_QNet class initializes two linear layers 
and implements the forward pass, while the QTrainer class handles the optimization and training steps, 
calculating the loss and updating the model's parameters using the Adam optimizer. The module also 
provides functionality to save the trained model's parameters to a specified file. Overall, this 
setup facilitates the training of an agent in a Q-learning framework.
"""

import torch  # To import the PyTorch library for deep learning tasks.
import torch.nn as nn  # To import the neural network module from PyTorch.
import torch.optim as optim  # To import the optimization algorithms from PyTorch.
import torch.nn.functional as F  # To import functional operations like activation functions.
import os  # To import the os module for interacting with the operating system, especially for file and directory operations.

# To define a neural network class for a linear Q-network.
class Linear_QNet(nn.Module):
    # To initialize the neural network with input size, hidden layer size, and output size.
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # To call the constructor of the parent class (nn.Module).
        # To create the first linear layer that connects input features to hidden units.
        self.linear1 = nn.Linear(input_size, hidden_size)
        # To create the second linear layer that connects hidden units to output actions.
        self.linear2 = nn.Linear(hidden_size, output_size)

    # To define the forward pass of the network, processing the input through the layers.
    def forward(self, x):
        x = F.relu(self.linear1(x))  # To apply the ReLU activation function to the output of the first layer.
        x = self.linear2(x)  # To get the final output from the second layer.
        return x  # To return the output of the network.

    # To save the model's parameters to a file.
    def save(self, file_name='model.pth'):
        model_folder_path = './model'  # To specify the folder where the model will be saved.
        if not os.path.exists(model_folder_path):  # To check if the folder exists.
            os.makedirs(model_folder_path)  # To create the folder if it does not exist.

        file_name = os.path.join(model_folder_path, file_name)  # To create the full path for the model file.
        torch.save(self.state_dict(), file_name)  # To save the model's state dictionary to the specified file.

# To define a class for training the Q-learning model.
class QTrainer:
    # To initialize the trainer with a model, learning rate (lr), and discount factor (gamma).
    def __init__(self, model, lr, gamma):
        self.lr = lr  # To store the learning rate for the optimizer.
        self.gamma = gamma  # To store the discount factor for future rewards.
        self.model = model  # To store the reference to the model being trained.
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # To initialize the Adam optimizer with the model's parameters.
        self.criterion = nn.MSELoss()  # To initialize the loss function as Mean Squared Error Loss.

    # To perform a training step given the current state, action taken, reward received, next state, and whether the game has ended.
    def train_step(self, state, action, reward, next_state, game_end):
        # To convert inputs to PyTorch tensors of appropriate types.
        state = torch.tensor(state, dtype=torch.float)  # To create a tensor for the current state.
        next_state = torch.tensor(next_state, dtype=torch.float)  # To create a tensor for the next state.
        action = torch.tensor(action, dtype=torch.long)  # To create a tensor for the action taken.
        reward = torch.tensor(reward, dtype=torch.float)  # To create a tensor for the reward received.

        # To check if the state tensor has a single dimension (batch size of 1).
        if len(state.shape) == 1:
            # To add an extra dimension to the state tensor to make it compatible with batch processing.
            state = torch.unsqueeze(state, 0)  
            next_state = torch.unsqueeze(next_state, 0)  
            action = torch.unsqueeze(action, 0)  
            reward = torch.unsqueeze(reward, 0)  
            game_end = (game_end,)  # To ensure game_end is a tuple for consistent indexing.

        # To get the predicted Q-values for the current state from the model.
        pred = self.model(state)

        target = pred.clone()  # To create a copy of the predicted Q-values for modification during training.
        # To iterate over each game end status to update the target values accordingly.
        for idx in range(len(game_end)):
            Q_new = reward[idx]  # To initialize the new Q-value with the reward received.
            if not game_end[idx]:  # To check if the game has not ended for the current index.
                # To update Q_new with the reward plus the discounted maximum predicted Q-value for the next state.
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # To update the target Q-value for the action taken based on the new Q-value calculated.
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # To reset the gradients of the optimizer for the current training step.
        self.optimizer.zero_grad()  
        # To calculate the loss between the predicted and target Q-values.
        loss = self.criterion(target, pred)  
        loss.backward()  # To perform backpropagation to compute gradients.

        self.optimizer.step()  # To update the model's parameters based on the computed gradients.
