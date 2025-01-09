# Neural network for flappy-bird game using python

## Showcase

``` training.py ``` showcase

https://github.com/user-attachments/assets/d2085a19-acc3-4507-8ad5-ee18627b85e8

## Installation
1. Ensure you have Python and pygame installed on your system (Python 3.6 or higher).
2. Clone this repository to your local machine using the following command:
3. Navigate to the project directory:
        ``` cd Flappy-Bird-AI ```
4. Install the required dependencies. It is recommended to set up a virtual environment before installing the dependencies:
        ``` pip install -r requirements.txt ```

## Running
1. To play the game manually, run the `main.py` script:
       ``` python main.py ```

2. To watch the AI play the game, run the `AI_mode.py` script:
       ``` python AI_mode.py ```

## Training the AI
Here's how to run the training:
1. Run the `training.py` script:

        python training.py
2. The training process will start, and you will see each generation's progress being printed to the console. The script will keep training until a bird successfully reaches a score of 100 or more.
3. Once a successful bird is found, the winning neural network's genome will be saved as `winner_genome.pkl`. This file contains the genetic information of the neural network that achieved the highest score during the training process.
4. You can then use this winner_genome.pkl file to observe the AI playing the game automatically by running the `AI_mode.py` script.

Feel free to modify the parameters and settings in `config-feedforward.txt` to experiment with the training process. You can adjust parameters like population size, mutation rate, and others to see how they impact the AI's learning.

## How the NEAT AI works
NEAT (NeuroEvolution of Augmenting Topologies) is a method of evolving artificial neural networks. In this project, the NEAT algorithm is utilized to train an AI to play the Flappy Bird game.

1. Initialization: The algorithm starts with a population of random neural networks (genomes) that control the birds.
2. Evaluation: Each bird's performance is scored using a fitness function, based on how far it travels, how many pipes it passes, etc.
3. Selection and Reproduction: The best-performing genomes are selected to create the next generation. They undergo mutations and crossover to improve performance.
4. Iteration: This process repeats for many generations until a bird reaches a score of 100 or more.
5. Saving the Best Genome: The best-performing genome is saved as winner_genome.pkl and used by AI_mode.py to demonstrate the AI playing the game automatically.

## resources
- https://realpython.com/python-ai-neural-network/
- https://vitalab.github.io/article/2017/05/29/geneticNN.html
