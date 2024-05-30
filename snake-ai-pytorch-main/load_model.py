import torch

from agent import Agent, SnakeGameAI
from model2 import Linear_QNet, QTrainer


model = Linear_QNet(11, 256, 3)
model.load_state_dict(torch.load('./model2/model2.pth'))


game = SnakeGameAI()
agent = Agent(model=model)


while True:
    state_old = agent.get_state(game)

    final_move = agent.get_action(state_old)

    reward, done, score = game.play_step(final_move)

    if done:
        game.reset()