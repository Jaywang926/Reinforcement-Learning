import gymnasium as gym
import numpy as np
from collections import defaultdict
import random

# Initialize environment
env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode=None)

### YOUR Q-LEARNING CODE BEGINS

num_episodes = 20000
alpha = 0.01  
gamma = 1
start_epsilon = 1
min_epsilon = 0.1

Q = defaultdict(lambda: np.zeros(env.action_space.n))

def get_action(state, epsilon, episode):

    if random.random() < epsilon:
        return env.action_space.sample()  
    else:
        return np.argmax(Q[state]) 
     
print(f"Training {num_episodes} episodes...")

def iterate(state, action, next_state, reward, done):
    next_action = np.argmax(Q[next_state])
    temp_diff = reward + gamma * Q[next_state][next_action] * (not done) - Q[state][action]
    Q[state][action] += alpha * temp_diff
    state = next_state
    return

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    epsilon = max(min_epsilon, start_epsilon - episode/num_episodes) #epsilon decay
    while not done:
        action = get_action(state, start_epsilon, episode)
        next_state, reward, terminated, truncated, info = env.step(action)
        iterate(state, action, next_state, reward, done)
        done = terminated or truncated


num_games = 10000
wins = 0
draws = 0
losses = 0

for i in range(num_games):
    state, info = env.reset()
    done = False
    
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

    if reward > 0:
        wins += 1
    elif reward == 0:
        draws += 1
    else:
        losses += 1

print(f"Simulated {num_games} games:")
print(f"Wins: {wins} ({wins / num_games:.2%})")
print(f"Draws: {draws} ({draws / num_games:.2%})")
print(f"Losses: {losses} ({losses / num_games:.2%})")