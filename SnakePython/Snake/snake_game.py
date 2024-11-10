"""
This code implements a Snake game using the Pygame library, featuring a simple AI for movement control. The game initializes the Pygame environment, setting up the display, colors, and fonts. 

The `Direction` enumeration defines the four possible movement directions (right, left, up, down). The `Point` class, defined using `namedtuple`, represents the coordinates of the snake and food on the screen. 

The `Snake` class encapsulates all game logic, including the initialization of game components, rendering the user interface, managing the snake's movement, checking for collisions, and updating the game state. 

The snake starts with a fixed length and grows when it consumes food, which is randomly placed on the screen. The game continues until the snake collides with itself or the game boundaries. The score is displayed on the screen and updates as the snake consumes food. 

**Keyboard Controls:**
- Arrow keys (↑, ↓, ←, →) are used to control the direction of the snake:
- Up Arrow: Move up
- Down Arrow: Move down
- Left Arrow: Move left
- Right Arrow: Move right

The game runs in a loop, handling user input and refreshing the display at a set frame rate.
"""


import pygame  # To import the Pygame library, which is used for game development
import random  # To import the random library, which will help place the food at random positions
from enum import Enum  # To import Enum for creating a set of symbolic names for directions
from collections import namedtuple  # To import namedtuple for defining a simple Point structure

# To initialize the Pygame library and set up its modules
pygame.init()

# To set up a font for rendering text (e.g., score display). 
# The font file 'arial.ttf' and size 25 are specified here.
font = pygame.font.Font('arial.ttf', 25)

# To define the four possible directions that the snake can move in using an enumeration.
# Enum values are assigned for each direction, which will be used later to control the snake's movement.
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# To define a Point class using namedtuple to store x and y coordinates.
# This is useful for representing positions on the game screen, such as the snake's position and food position.
Point = namedtuple('Point', 'x, y')

# To define color values in RGB format for different game elements like snake, food, and background.
WHITE = (255, 255, 255)  # Color for score text
RED = (255, 0, 0)        # Color for food
BLUE = (0, 0, 255)       # Color for snake's body
BLUEE = (0, 100, 255)    # Slightly darker blue for the inner part of the snake's body for a layered look
BLACK = (0, 0, 0)        # Color for the game background

# To set up the size of each block that makes up the snake and the food.
# BLOCK_SIZE will be used to control the snake's size and movement increment.
BLOCK_SIZE = 20

# To define the speed of the game, which controls how quickly the snake moves each step.
SPEED = 10

# To define the main Snake class, which contains all attributes and methods for handling game logic.
# This includes initialization, drawing the UI, moving the snake, checking collisions, and updating the game.
class Snake:
    # To initialize the game with screen width, height, and initial setup for game components.
    # width and height parameters define the display dimensions.
    def __init__(self, width=640, height=480):
        # To set the display width and height to specified values or defaults
        self.width = width
        self.height = height

        # To create a display surface of defined size (width, height) for rendering game graphics
        self.display = pygame.display.set_mode((self.width, self.height))

        # To set the window title to "Snek!!!"
        pygame.display.set_caption('Snek!!!')

        # To create a clock object to control the frame rate of the game loop
        self.clock = pygame.time.Clock()

        # To initialize the snake's movement direction to the right
        self.direction = Direction.RIGHT

        # To initialize the snake's head position at the center of the screen
        self.head = Point(self.width / 2, self.height / 2)

        # To create an initial snake body with three blocks, starting with the head and extending left
        # Each block is represented as a Point, spaced by BLOCK_SIZE
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y), 
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        # To set the initial score to zero
        self.score = 0

        # To initialize the food position; set to None before placing it on the screen
        self.food = None

        # To call the helper method that places food at a random position on the screen
        self._place_food()

    # To place food at a random position on the screen that does not overlap with the snake's body
    def _place_food(self):
        # To calculate a random x and y position within screen boundaries and snap to BLOCK_SIZE grid
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)  # To set the food position using the random coordinates

        # To check if the food position overlaps with the snake's body, recursively call _place_food if it does
        if self.food in self.snake:
            self._place_food()

    # To handle each step of the game (one frame), including handling user input, moving the snake, and checking for collisions
    def play_step(self):
        # To handle user events, such as quitting or pressing keys to change the snake's direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()  # To exit the game

            # To check for keyboard input to control the direction of the snake
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN

        # To update the snake's position by moving in the current direction
        self._move(self.direction)

        # To add the new head position to the beginning of the snake body list
        self.snake.insert(0, self.head)

        # To check if the snake has collided with itself or the walls
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score  # To end the game if there's a collision

        # To check if the snake's head has reached the food position
        if self.head == self.food:
            # To increase the score by 1 and place new food if the snake eats the food
            self.score += 1
            self._place_food()
        else:
            # To remove the last block of the snake if no food is eaten (snake moves without growing)
            self.snake.pop()

        # To update the game graphics and show the current state of the snake and food
        self._update_ui()

        # To control the game's frame rate
        self.clock.tick(SPEED)

        # To return the game_over status and current score at the end of each step
        return game_over, self.score

    # To check if the snake has collided with the wall or itself
    def _is_collision(self):
        # To check if the snake's head is outside the screen boundaries
        if (self.head.x > self.width - BLOCK_SIZE or self.head.x < 0 or 
            self.head.y > self.height - BLOCK_SIZE or self.head.y < 0):
            return True  # Return True if collision with wall

        # To check if the snake's head has collided with any part of its body (excluding the head itself)
        if self.head in self.snake[1:]:
            return True  # Return True if collision with itself

        # To return False if no collision is detected
        return False

    # To update and render the user interface, including the snake, food, and score
    def _update_ui(self):
        # To fill the display with a black background
        self.display.fill(BLACK)

        # To draw each segment of the snake on the screen
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))  # Outer block
            pygame.draw.rect(self.display, BLUEE, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))  # Inner block for effect

        # To draw the food as a red square on the screen
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # To display the score on the top left corner
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])

        # To update the screen with the latest drawings
        pygame.display.flip()

    # To move the snake in the specified direction by updating the position of the head
    def _move(self, direction):
        # To retrieve current x and y positions of the snake's head
        x = self.head.x
        y = self.head.y

        # To change the head's position based on the current direction
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        # To set the new head position with updated x and y values
        self.head = Point(x, y)

# To start the game and create an instance of the Snake class
if __name__ == '__main__':
    game = Snake()

    # To run the main game loop, calling play_step repeatedly until the game is over
    while True:
        game_over, score = game.play_step()

        # To break out of the game loop if game_over is True
        if game_over:
            break

    # To display the final score when the game ends
    print(f'Final Score: {score}')

    # To quit Pygame properly
    pygame.quit()
