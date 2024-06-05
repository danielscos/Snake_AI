# Snake AI Pytorch
This is a simple snake game with AI implemented using Pytorch. The snake is controlled by the AI which is trained using a simple feedforward neural network. The game is implemented using Pygame.
I learned how to do this and got the idea from a youtube tutorial made by FreeCodeCamp, link below.
## Requirements
- Python 3.6 or higher
- Pytorch
- Pygame
- Numpy
- Matplotlib

## Installation
1. Clone the repository
```bash
git clone https://github.com/danielscos/Snake_AI.git
```
2. Install the required packages
```bash
pip install -r requirements.txt
```
you successfully installed the required packages.

## Usage
If you run snake_game_human.py you can play the game yourself. If you run agent.py the AI will play the game, the game will restart when the snake dies. The progression of the neural network can be seen on the matplotlib graph. The graph shows the score and the number of games of the neural network.
As you can see, the neural network is learning to play the game, as the number of games increases the neural network becomes stronger.
One way i like to call the "number of games" is the "generation" of the neural network, because each game the neural network reborns stronger then before, just like a generation of a species.

## Thank you
I hope you enjoy the game and the AI. If you have any questions or suggestions, feel free to contact me through this github repository.

## Idea from: https://youtu.be/L8ypSXwyBds?si=IEBaJv1QuiMJ4eoW

## Project Made By - Daniel Grosso


[![Alternate Text](./Screenshot%202024-05-31%20005555.png)](https://youtu.be/SFEREJI355E "Showcase / Tutorial Video")
```