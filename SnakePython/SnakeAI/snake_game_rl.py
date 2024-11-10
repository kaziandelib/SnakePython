"""
This code implements a Snake AI game using the Pygame library. It initializes the game environment, including the game window, colors, and font settings. The `SnakeAI` class manages the game state, including the snake's movement, food placement, score tracking, and collision detection. The game operates in a grid where the snake moves in four possible directions (right, left, up, down) and grows in length when it consumes food. The AI can make decisions based on its current state to navigate the game efficiently. The game runs in a loop, updating the display and handling user inputs until the game ends.
"""



import pygame  # To import the Pygame library, which is essential for creating video games and multimedia applications in Python.
import random  # To import the random module, which allows for generating random numbers; useful for placing food in the game randomly.
from enum import Enum  # To import the Enum class, which allows for the creation of enumerated types; this helps define the possible directions of the snake.
from collections import namedtuple  # To import namedtuple from the collections module, which provides a way to create tuple-like objects with named fields; useful for representing points in the game.
import numpy as np  # To import the NumPy library, which is used for numerical computing; here, it helps with handling arrays for the snake's movement decisions.

# To initialize the Pygame library, which is necessary before using any Pygame functionality.
pygame.init()
# To create a font object from the specified font file ('arial.ttf') with a size of 25, used for rendering text in the game.
font = pygame.font.Font('arial.ttf', 25)

# To define an enumeration for the four possible directions the snake can move, making the code more readable and manageable.
class Direction(Enum):
    RIGHT = 1  # To represent the right direction.
    LEFT = 2   # To represent the left direction.
    UP = 3     # To represent the upward direction.
    DOWN = 4   # To represent the downward direction.

# To create a Point class using namedtuples for easy representation of coordinates (x, y) in the game.
Point = namedtuple('Point', 'x, y')

# To define RGB color values for use in the game; these colors will be used for the snake, food, and background.
# To set the color white (RGB: 255, 255, 255).
WHITE = (255, 255, 255)
# To set the color red (RGB: 200, 0, 0).
RED = (200, 0, 0)
# To set the color blue (RGB: 0, 0, 255).
BLUE = (0, 0, 255)
# To set a lighter shade of blue (RGB: 0, 100, 255).
BLUEE = (0, 100, 255)
# To set the color black (RGB: 0, 0, 0).
BLACK = (0, 0, 0)

# To define the size of each block in the game (snake and food); each block will be a square of this size.
BLOCK_SIZE = 20
# To define the speed of the game; the game will run at this speed, determining how fast the game updates.
SPEED = 4_000

# To create a class representing the Snake AI, which will manage the game state, snake movement, and food placement.
class SnakeAI:
    # To initialize the SnakeAI class with a specified width and height for the game window.
    def __init__(self, width=640, height=480):
        self.width = width  # To store the width of the game window.
        self.height = height  # To store the height of the game window.
        # To initialize the display for the game using the specified width and height.
        self.display = pygame.display.set_mode((self.width, self.height))
        # To set the title of the game window.
        pygame.display.set_caption('Snake')
        # To create a clock object to control the game's frame rate.
        self.clock = pygame.time.Clock()
        # To call the reset method to initialize the game state.
        self.reset()

    # To reset the game state to start a new game.
    def reset(self):
        # To initialize the direction of the snake to the right.
        self.direction = Direction.RIGHT

        # To set the initial position of the snake's head at the center of the screen.
        self.head = Point(self.width / 2, self.height / 2)
        # To create the initial body of the snake with three segments.
        self.snake = [
            self.head, 
            Point(self.head.x - BLOCK_SIZE, self.head.y), 
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        self.score = 0  # To initialize the score to zero.
        self.food = None  # To initialize the food position as None; it will be placed later.
        self._place_food()  # To call the method to place the first food item on the screen.
        self.frame_iteration = 0  # To initialize the frame iteration counter.

    # To randomly place food on the game board.
    def _place_food(self):
        # To generate random coordinates for the food within the boundaries of the game window.
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)  # To create a Point object for the food's position.
        # To ensure food does not spawn on the snake's body; if it does, place food again.
        if self.food in self.snake:
            self._place_food()

    # To handle one step of the game, including movement and collision detection.
    def play_step(self, action):
        self.frame_iteration += 1  # To increment the frame iteration counter.

        # To check for any events (like quitting the game).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()  # To quit Pygame.
                quit()  # To exit the program.
        
        self._move(action)  # To update the snake's position based on the chosen action.
        self.snake.insert(0, self.head)  # To add the new head position to the snake's body.
        
        reward = 0  # To initialize the reward for this step.
        game_over = False  # To flag whether the game is over.

        # To check for collision with the walls or if the snake has grown too long.
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True  # To set the game over flag.
            reward = -10  # To assign a negative reward for colliding.
            return reward, game_over, self.score  # To return the reward, game over status, and score.

        # To check if the snake's head has collided with the food.
        if self.head == self.food:
            self.score += 1  # To increase the score by 1.
            reward = 10  # To assign a positive reward for eating food.
            self._place_food()  # To place a new food item on the screen.
        else:
            self.snake.pop()  # To remove the last segment of the snake if it hasn't eaten food.
        
        self._update_ui()  # To update the game's user interface.
        self.clock.tick(SPEED)  # To control the game speed by setting the frame rate.
        return reward, game_over, self.score  # To return the reward, game over status, and score.

    # To check for collisions with walls or the snake's body.
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head  # To set the point to check for collisions to the snake's head if not provided.
        
        # To check if the point is outside the boundaries of the game window.
        if pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
            return True  # To return True if the point collides with a wall.
        
        # To check if the point collides with the snake's body (excluding the head).
        if pt in self.snake[1:]:
            return True  # To return True if the point collides with the body.

        return False  # To return False if no collision is detected.

    # To update the user interface of the game (rendering the snake, food, and score).
    def _update_ui(self):
        self.display.fill(BLACK)  # To fill the display with the background color (black).

        # To draw each segment of the snake on the display.
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))  # To draw the main part of the snake.
            pygame.draw.rect(self.display, BLUEE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))  # To draw the inner part of the snake for a 3D effect.

        # To draw the food on the display.
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # To render the score as text and display it in white at the top left corner.
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])  # To place the score text on the display.
        pygame.display.flip()  # To update the display with the newly drawn elements.

    # To handle the movement of the snake based on the provided action.
    def _move(self, action):
        # To define the clockwise order of directions to simplify direction changes.
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)  # To get the current index of the snake's direction.

        # To determine the new direction based on the action taken by the AI.
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # To continue in the current direction.
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # To turn right (clockwise).
            new_dir = clock_wise[next_idx]  # To update the direction to the next clockwise direction.
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # To turn left (counter-clockwise).
            new_dir = clock_wise[next_idx]  # To update the direction to the next counter-clockwise direction.

        self.direction = new_dir  # To set the snake's new direction.

        # To update the snake's head position based on the new direction.
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE  # Move the head right.
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE  # Move the head left.
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE  # Move the head down.
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE  # Move the head up.

        self.head = Point(x, y)  # To update the head position with the new coordinates.
