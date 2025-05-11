import gymnasium as gym
from collections import defaultdict

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None) #initialization


# Q 2.2

def get_T_R_with_random_policy():
    successful_transitions = defaultdict(lambda: defaultdict(float))
    total_transitions = defaultdict(int)
    rewards = defaultdict(lambda: defaultdict(float))
    T = defaultdict(lambda: defaultdict(float))
    R = defaultdict(lambda: defaultdict(float))
    for i in range(1000):
        state, info = env.reset()
        done = False
        while not done:
    
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            successful_transitions[state, action][next_state] += 1
            #print(successful_transitions[state, action][next_state])
            total_transitions[state, action] += 1
            rewards[state, action][next_state] += reward
        
            # if rewards[state, action][next_state] > 0:
            #     print(state, action, next_state, rewards[state, action][next_state])
            done = terminated or truncated
            state = next_state
        for state, action in successful_transitions:
            for next_state in successful_transitions[state, action]:
                T[state, action][next_state] = successful_transitions[state, action][next_state]/total_transitions[state, action]
                R[state, action][next_state] = rewards[state, action][next_state]/successful_transitions[state, action][next_state]
    return T, R


#Q 2.3 and 2.4
def has_converged(change):
    if change < 0.0000001:
        return True
    return False

def get_V_and_policy(T, R):
    V = defaultdict(float)
    policy = defaultdict(float)
    change = float('inf')
    gamma = 0.9
    states = set()
    actions = {0, 1, 2, 3}
    for state, action in T:
        states.add(state)

    states.add(15)

    while not has_converged(change):
        change = 0
        for state in states:
            old_v = V[state]
            best_value = 0
            for action in actions:
                sum_value = 0
                for next_state in T[state, action]:
                    Tsas = T[state, action][next_state]
                    Rsas = R[state, action][next_state]
                    sum_value += Tsas * (Rsas + gamma * V[next_state])
                
                if sum_value > best_value:

                    best_value = sum_value
                    policy[state] = action
                    V[state] = best_value
            
            diff = abs(V[state] - old_v)
            if change < diff:
                change = diff
            

    return V, policy

T, R = get_T_R_with_random_policy()
V, policy = get_V_and_policy(T, R)

print('Optimal Values:')
for state, value in dict(sorted(V.items())).items():
    print(f"State: {state}, Value: {value}")
print('\n')

print('Optimal Policies:')
for state, action in dict(sorted(policy.items())).items():
    print(f"State: {state}, Action: {action}")
print('\n')

#Q 2.5
num_games = 1000
successes = 0
fails = 0
for i in range(num_games):
    state, info = env.reset()
    done = False
    
    while not done:
        action = policy[state]
        #action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

    if reward > 0:
        successes += 1
    else:
        fails += 1
print(f"Simulated {num_games} games:")
print(f"Successes: {successes} ({successes / num_games:.2%})")
print(f"Fails: {fails} ({fails / num_games:.2%})")
env.close()