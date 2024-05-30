import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from game import SnakeGameAI




def train():
    game = SnakeGameAI()
    model = QNet()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    memory = deque(maxlen=10000)
    batch_size = 64
    epochs = 1000
    steps = 0

    for epoch in range(epochs):
        state = game.reset()
        done = False

        while not done:
            steps += 1

            # Select action
            if np.random.rand() < 0.1:
                action = np.random.randint(3)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float)
                action = torch.argmax(model(state_tensor)).item()

            # Perform action
            next_state, reward, done = game.step(action)

            # Store in memory
            memory.append((state, action, reward, next_state, done))

            # Train the model
            if steps % 10 == 0 and len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

                state_batch = torch.tensor(state_batch, dtype=torch.float)
                action_batch = torch.tensor(action_batch, dtype=torch.long)
                reward_batch = torch.tensor(reward_batch, dtype=torch.float)
                next_state_batch = torch.tensor(next_state_batch, dtype=torch.float)
                done_batch = torch.tensor(done_batch, dtype=torch.bool)

                current_q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
                next_q_values = model(next_state_batch).max(1)[0]
                target_q_values = reward_batch + 0.99 * next_q_values * (~done_batch)

                loss = criterion(current_q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        print(f'Epoch: {epoch}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train()