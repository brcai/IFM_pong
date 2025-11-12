import pygame
import random
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import random

from state_nn import ExplicitNeuralNetwork


def int_to_binary_list(n, bits = 8):
    twos_complement = n & (2**bits - 1)
    return [int(bit) for bit in format(twos_complement, f'0{bits}b')]


seed = 0

torch.manual_seed(seed)
random.seed(seed) 

l = 20
batch_size = 32
device = 'cpu'

n_neurons = 200
p_connect = 0.2

model = ExplicitNeuralNetwork(num_neurons=n_neurons, num_inputs=24, num_outputs=3, connection_prob=p_connect, device=device).to(device)
model.load_state_dict(torch.load("save/final_model_200_0.2_32_20_4.pth"))
model.eval()

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle settings
PADDLE_WIDTH = 50
PADDLE_HEIGHT = 10
PADDLE_SPEED = 7

# Ball settings
BALL_SIZE = 15
BALL_SPEED_X = 4
BALL_SPEED_Y = -4

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Paddle Ball Game")

# Initialize paddle
paddle = pygame.Rect(SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2, SCREEN_HEIGHT - 40, PADDLE_WIDTH, PADDLE_HEIGHT)

# Initialize ball
ball = pygame.Rect(random.choice([i*10 for i in range(3,17)]), random.choice([i*10 for i in range(3,25)]), BALL_SIZE, BALL_SIZE)
ball_dx = BALL_SPEED_X
ball_dy = BALL_SPEED_Y

# Game loop
clock = pygame.time.Clock()

d = []

idx = 0
states = None
neuron_outputs = None

with torch.no_grad():
    while idx < 20000000000:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        x = ball.x
        y = ball.y
        p_x = paddle.x
        p_y = paddle.y
        press = 0

        tmp = []
        tmp.extend(int_to_binary_list(int(x) + 2))
        tmp.extend(int_to_binary_list(int(y) + 2))
        tmp.extend(int_to_binary_list(int(p_x) + 2))
        x_input = torch.tensor(tmp, dtype=torch.float32).reshape(1,-1,24)
    
    
        #pred, s, o = model(x_input*0, states, neuron_outputs)
        pred, s, o = model(x_input, states, neuron_outputs)
        states = s
        neuron_outputs = o
        ctr = torch.argmax(pred).item()
        if ctr == 0:
            press = 1
        elif ctr == 1:
            press = -1

        if press == -1 and paddle.left > 0:
            paddle.x -= PADDLE_SPEED
        if press == 1 and paddle.right < SCREEN_WIDTH:
            paddle.x += PADDLE_SPEED

        # Ball movement
        ball.x += ball_dx
        ball.y += ball_dy

        # Ball collision with walls
        if ball.left <= 0 or ball.right >= SCREEN_WIDTH:
            ball_dx *= -1
        if ball.top <= 0:
            ball_dy *= -1
        if ball.bottom >= SCREEN_HEIGHT:
            print("Game Over!")
            ball = pygame.Rect(random.choice([i*10 for i in range(3,17)]), random.choice([i*10 for i in range(3,25)]), BALL_SIZE, BALL_SIZE)
            ball_dx = BALL_SPEED_X
            ball_dy = BALL_SPEED_Y
            #pygame.quit()
            #sys.exit()

        # Ball collision with paddle
        if ball.colliderect(paddle) and ball_dy > 0:
            ball_dy *= -1

        # Drawing
        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, paddle)
        pygame.draw.ellipse(screen, WHITE, ball)

        # Refresh screen
        pygame.display.flip()
        #clock.tick(60)
        idx += 1



    
    
