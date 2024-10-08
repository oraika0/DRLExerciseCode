# import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import pygame
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Super Mario Bros Control")
scale_factor = 2.5# Create the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = True
action = 0  # Initialize action to idle

# Create a clock to control the frame rate
clock = pygame.time.Clock()

while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            print("keydown")
            if event.key == pygame.K_UP:
                action = 5  # Corresponding action for UP
            elif event.key == pygame.K_DOWN:
                action = 10  # Corresponding action for DOWN
            elif event.key == pygame.K_LEFT:
                action = 6  # Corresponding action for LEFT
            elif event.key == pygame.K_RIGHT:
                action = 1  # Corresponding action for RIGHT
            elif event.key == pygame.K_a:
                action = 7
            elif event.key == pygame.K_s:
                action = 2
            elif event.key == pygame.K_z:
                action = 8
            elif event.key == pygame.K_x:
                action = 3
            
        elif event.type == pygame.KEYUP:
            action = 0  # Reset action on key release (no action)

    # Reset the environment if done
    if done:
        state = env.reset()
        print("Game Reset")

    # Step the environment
    state, reward, done, info = env.step(action)
    print('x_pos :',info['x_pos'])

    frame = env.render('rgb_array')
    frame = np.fliplr(frame)  # Flip the frame vertically
    frame = np.rot90(frame)
    # Scale the frame
    frame = pygame.surfarray.make_surface(frame)
    scaled_frame = pygame.transform.scale(frame, (frame.get_width() * scale_factor, frame.get_height() * scale_factor))

    # Render the environment
    screen.blit(scaled_frame, (0, 0))
    pygame.display.update()  # Update the display

    # Control the frame rate
    clock.tick(30)  # Adjust this value to control the speed of the game

