# Student name: Terence Chu
# Student unmber: 301220117

import gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

# Create environment
env = gym.make('LunarLander-v2', render_mode='rgb_array')

# There are 8 state variables 
state_size = env.observation_space.shape[0]
print(state_size)

# There are 4 possible actions 
# - Do nothing
# - Fire left engine
# - Fire right engine
# - Fire down engine
action_size = env.action_space.n
print(action_size)

# Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(state_size,)),
    Dense(256, activation='relu'),
    Dense(action_size, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

model.summary()

# Define epsilon greedy policy
# If a random number is less than epsilon, explore with a random action
def epsilon_greedy_policy(state, epsilon, model, action_size):
    if np.random.rand() < epsilon:
        # Explore with random action
        return np.random.randint(action_size)
    else:
        # Exploit (greedy action)
        # Reshape state array to (1, 8) - i.e., 1 batch of 8 states
        # Predict the Q value for each possible action with the current state
        # [0] selects the first element, which represents the predicted award for a certain action
        state = np.reshape(state, (1, -1))
        Q_values = model.predict(state, verbose=0)[0]
        return Q_values.argmax()  
    
# Reset the environment
state = env.reset()

# Define the hyperparameters

# Number of episodes (number of completed runs - success, failure, or max steps reached)
num_episodes = 750

# Maximum number of steps per episode
max_steps_per_episode = 500

# Discount factor (gamma)
# 0.99 emphasizes future rewards
discount_factor = 0.99  

# Set initial exploration probability to 1.0 (full exploration, no exploitation)
epsilon = 1.0 

# Full exploitation
epsilon_min = 0.01

# Rate of decay for epsilon (decay by 0.5% per episode)
epsilon_decay = 0.995

batch_size = 128

# Replay memory holds the past experiences of the agent
# A size of 100000 allows the agent to learn from a diverse number of experiences
replay_memory_size = 100000

# replay_buffer holds the actual experiences
# Set its size to replay_memory_size
# Any new experience beyond 100000 will cause the oldest experience to be removed
replay_buffer = deque(maxlen=replay_memory_size)

def train_model(model, replay_buffer, batch_size, discount_factor):
    
    # The replay buffer populates by 1 experience per step
    # Only train the model after 64 steps (to ensure diverse experience)
    if len(replay_buffer) < batch_size:
        return
    
    # Randomly 64 samples from the replay buffer
    batch = random.sample(replay_buffer, batch_size)

    # Extract 5 lists
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert states to numpy arrays
    states = np.array(states)
    next_states = np.array(next_states)

    # Predict Q-values for current and next states
    current_q_values = model.predict(states, verbose=0)
    next_q_values = model.predict(next_states, verbose=0)

    # Calculate the target Q-value
    for i, (state, action, reward, next_state, done) in enumerate(batch):
        if done: # Episode is complete - no next state to consider, assign it the final reward
            target_q_value = reward
        else: # Calculate with Bellman equation 
            target_q_value = reward + discount_factor * np.max(next_q_values[i])
        
        # Update the Q-value for the chosen action
        current_q_values[i, action] = target_q_value

    # Train the model on the current batch of experiences
    # Minimize difference between the predicted and target Q values
    model.fit(states, current_q_values, verbose=0)

episode_rewards = []
step_counts = []
frames = []

# Loop through each episode
for episode in range(num_episodes):
    # Reset the environment
    state, _ = env.reset()
    total_reward = 0
    steps = 0

    # Loop through each step
    for step in range(max_steps_per_episode):
        if episode % 25 == 0:
            # Capture the frame
            frame = env.render() 
            
            # Add episode number to the frame
            # Code adapted from https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
            frame = cv2.putText(
                frame,
                f"Episode: {episode + 1}",
                (10, 30), # Location (X and Y coordinates)
                cv2.FONT_HERSHEY_SIMPLEX, # Font
                1,  # Font size
                (255, 255, 255),  # Font color (white)
                2,  # Thickness
                cv2.LINE_AA  # Draw with anti-aliased lines to smooth out edges
            )
            
            # Store the frame
            frames.append(frame)  

        # Select an action with the epsilon greedy policy
        # Exploration = random action
        # Exploitation = carry out the action that yields the highest predicted Q-value
        action = epsilon_greedy_policy(state, epsilon, model, action_size)

        # Execute the action and store the next_state, reward, and done (whether episode has ended)
        next_state, reward, done, _, _ = env.step(action)

        # Add the current experience to the buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Update state for the next step
        state = next_state
        
        total_reward += reward
        steps += 1

        # Train the model using a random batch of experiences from the replay buffer
        train_model(model, replay_buffer, batch_size, discount_factor)

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    episode_rewards.append(total_reward)
    step_counts.append(steps)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()

# Save the video
video_filename = "lunar_lander.mp4"
imageio.mimsave(video_filename, frames, fps=30)
print(f"Video saved as {video_filename}")

cumulative_rewards = np.cumsum(episode_rewards)

# Plotting the cumulative rewards
plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Rewards")
plt.title("Cumulative Reward Over Episodes")
plt.legend()
plt.grid()
plt.show()

# Plotting the episode lengths
plt.figure(figsize=(10, 6))
plt.plot(step_counts)
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.title("Number of Steps Per Episode")
plt.legend()
plt.grid()
plt.show()