import pygame
import random
import sys

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
#ball = pygame.Rect(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, BALL_SIZE, BALL_SIZE)
ball = pygame.Rect(random.choice([i for i in range(50,150)]), random.choice([i for i in range(50,150)]), BALL_SIZE, BALL_SIZE)
ball_dx = BALL_SPEED_X
ball_dy = BALL_SPEED_Y

# Game loop
clock = pygame.time.Clock()

d = []

fp = open("d5.txt", 'w')

idx = 0

while idx < 20000:
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
    
    # Paddle movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle.left > 0:
        paddle.x -= PADDLE_SPEED
    if keys[pygame.K_RIGHT] and paddle.right < SCREEN_WIDTH:
        paddle.x += PADDLE_SPEED

    if keys[pygame.K_LEFT]:
        press = -1
    if keys[pygame.K_RIGHT]:
        press = 1

    print(idx, press)
    fp.write(f"{ball.x}\t{ball.y}\t{paddle.x}\t{paddle.y}\t{press}\n")

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
    clock.tick(60)
    idx += 1

fp.close()

    
    
