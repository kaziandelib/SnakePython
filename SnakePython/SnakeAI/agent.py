"""
This script implements a reinforcement learning agent for playing the Snake game using Q-learning. 
It utilizes PyTorch for building a neural network model that predicts the best actions based on the current state of the game. 
The agent maintains a memory of past experiences to improve its decision-making over time. 
The training process involves interacting with the game environment, storing experiences, and optimizing the model through a combination of short-term and long-term memory training.
The training scores are plotted to visualize the agent's performance improvement across games.
"""

import torch  # To import the PyTorch library for building and training neural networks.
import random  # To import the random module for generating random numbers, aiding in exploration strategies.
import numpy as np  # To import NumPy for numerical operations, particularly with arrays.
from collections import deque  # To import deque for maintaining a fixed-length memory for experiences.
from snake_game_rl import SnakeAI, Direction, Point  # To import necessary classes and enums from the snake_game_rl module.
from model import Linear_QNet, QTrainer  # To import the neural network model and trainer classes for Q-learning.
from helper import plot  # To import the plot function for visualizing training scores.

MAX_MEMORY = 100_000  # To define the maximum size of the memory to store experiences.
BATCH_SIZE = 1000  # To set the batch size for training samples drawn from memory.
learning_rate = 0.001  # To define the learning rate for the optimizer used in training.

class Agent:  # To define the Agent class that represents the reinforcement learning agent.
    def __init__(self):
        self.n_games = 0  # To initialize the number of games played by the agent.
        self.epsilon = 0  # To initialize the exploration factor for the epsilon-greedy strategy.
        self.gamma = 0.85  # To set the discount factor for future rewards.
        self.memory = deque(maxlen=MAX_MEMORY)  # To initialize a memory deque with a maximum size for storing experiences.
        self.model = Linear_QNet(11, 256, 3)  # To create an instance of the neural network model.
        self.trainer = QTrainer(self.model, lr=learning_rate, gamma=self.gamma)  # To create a QTrainer for training the model.

    def get_state(self, game):  # To define a method that retrieves the current state of the game.
        head = game.snake[0]  # To get the position of the snake's head.
        point_l = Point(head.x - 20, head.y)  # To calculate the point to the left of the head.
        point_r = Point(head.x + 20, head.y)  # To calculate the point to the right of the head.
        point_u = Point(head.x, head.y - 20)  # To calculate the point above the head.
        point_d = Point(head.x, head.y + 20)  # To calculate the point below the head.

        dir_l = game.direction == Direction.LEFT  # To check if the current direction is left.
        dir_r = game.direction == Direction.RIGHT  # To check if the current direction is right.
        dir_u = game.direction == Direction.UP  # To check if the current direction is up.
        dir_d = game.direction == Direction.DOWN  # To check if the current direction is down.

        state = [  # To define the current state representation as a list of binary features.
            # To assess the danger straight ahead.
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # To assess the danger to the right.
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # To assess the danger to the left.
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # To indicate the current movement direction.
            dir_l, dir_r, dir_u, dir_d,
            
            # To determine the position of food relative to the snake's head.
            game.food.x < game.head.x,  # To check if the food is to the left.
            game.food.x > game.head.x,  # To check if the food is to the right.
            game.food.y < game.head.y,  # To check if the food is above.
            game.food.y > game.head.y  # To check if the food is below.
        ]

        return np.array(state, dtype=int)  # To convert the state list into a NumPy array for processing.

    def remember(self, state, action, reward, next_state, game_end):  # To define a method to store experiences in memory.
        self.memory.append((state, action, reward, next_state, game_end))  # To add the experience tuple to memory.

    def train_long_memory(self):  # To define a method for training using the long-term memory.
        if len(self.memory) > BATCH_SIZE:  # To check if enough experiences are stored.
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # To randomly sample a batch from memory.
        else:
            mini_sample = self.memory  # To use the entire memory if it's smaller than the batch size.

        states, actions, rewards, next_states, game_ends = zip(*mini_sample)  # To unpack the sampled experiences.
        self.trainer.train_step(states, actions, rewards, next_states, game_ends)  # To train the model on the sampled experiences.

    def train_short_memory(self, state, action, reward, next_state, game_end):  # To define a method for training using a single experience.
        self.trainer.train_step(state, action, reward, next_state, game_end)  # To train the model on the provided experience.

    def get_action(self, state):  # To define a method that determines the next action to take based on the current state.
        self.epsilon = 80 - self.n_games  # To adjust the exploration rate based on the number of games played.
        final_move = [0, 0, 0]  # To initialize the final move as a zero vector representing no movement.

        if random.randint(0, 200) < self.epsilon:  # To explore randomly with a probability based on epsilon.
            move = random.randint(0, 2)  # To choose a random move (0, 1, or 2).
            final_move[move] = 1  # To update the final move vector with the chosen random action.
        else:  # To exploit the learned behavior by selecting the best action.
            state0 = torch.tensor(state, dtype=torch.float)  # To convert the state to a tensor for model input.
            prediction = self.model(state0)  # To get the model's predictions for the current state.
            move = torch.argmax(prediction).item()  # To find the index of the action with the highest predicted value.
            final_move[move] = 1  # To update the final move vector with the best action.

        return final_move  # To return the chosen action vector.


def train():  # To define the training loop for the agent.
    plot_scores = []  # To initialize a list for storing individual game scores.
    plot_mean_scores = []  # To initialize a list for storing mean scores over games.
    total_score = 0  # To initialize the total score accumulator.
    record = 0  # To initialize the record score.
    agent = Agent()  # To create an instance of the Agent class.
    game = SnakeAI()  # To create an instance of the SnakeAI class representing the game environment.
    
    while True:  # To run the training loop indefinitely.
        state_old = agent.get_state(game)  # To get the current state representation of the game.

        final_move = agent.get_action(state_old)  # To get the action the agent will take based on the current state.

        reward, game_end, score = game.play_step(final_move)  # To execute the action in the game and get the resulting state and reward.
        state_new = agent.get_state(game)  # To get the new state representation after the action.

        agent.train_short_memory(state_old, final_move, reward, state_new, game_end)  # To train the agent using the most recent experience.

        agent.remember(state_old, final_move, reward, state_new, game_end)  # To store the experience in memory for future training.

        if game_end:  # To check if the game has ended.
            game.reset()  # To reset the game state for the next round.
            agent.n_games += 1  # To increment the number of games played.
            agent.train_long_memory()  # To train the agent using experiences stored in memory.

            if score > record:  # To check if the current score is a new record.
                record = score  # To update the record score.
                agent.model.save()  # To save the current model if a new record is achieved.

            print(f'Game {agent.n_games}, Score: {score}, Record: {record}')  # To output the current game number, score, and record.

            plot_scores.append(score)  # To add the current score to the list of scores for plotting.
            total_score += score  # To accumulate the total score.
            mean_score = total_score / agent.n_games  # To calculate the mean score across all games played.
            plot_mean_scores.append(mean_score)  # To add the mean score to the list for plotting.
            plot(plot_scores, plot_mean_scores)  # To visualize the training scores.


if __name__ == "__main__":  # To ensure this block runs only if the script is executed directly.
    train()  # To call the training function to start the agent training process.
