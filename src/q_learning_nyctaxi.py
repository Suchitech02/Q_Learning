import gymnasium as gym
import numpy as np

def train_q_learning(episodes=5000):
    env = gym.make("Taxi-v3")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999

    for ep in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return Q

if __name__ == "__main__":
    Q = train_q_learning()
    print("Training complete!")
