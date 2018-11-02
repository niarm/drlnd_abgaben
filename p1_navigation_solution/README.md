### Solution for Deep Reinforcement Learning Nanodegree on Udacity - Project#1: "Navigation"
This is my solution to the first project in the "Udacity Deep Reinforcement Learning Nanodegree"

#### Project Details
This Project contains a unity-environment in which an Agent tries to learn to collect yellow bananas and avoid blue bananas while 
moving around a 3-dimensional room. 

The Environment has a State-Space of 37 dimensions, each representing the result of a raycast 
from the agent into the 3d-Environment.

The Agent can chose amongst 4 Actions: "move forward, move backwards, turn left, turn right"

The Environment is considered as solved, if the Agent can reach an average score over 13 in 100 consecutive Episodes


#### Getting Started
To run the main Notebook "main_solution.ipynb", please install:

1. pytorch 0.4
-- pip install torch

2. unityagents (you don't have to install unity3d to run the project!)
-- pip install unityagents

3. matplotlib
-- pip install matplotlib

#### Instructions
Follow the instructions inside the Jupyter-Notebook "main_solution.ipynb"

#### Credit
This Solution is based upon the implementation of an DQN(Deep-Q-Network)
by the udacity Team: https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution
and basically a copy of their code, adjusted to work with the Banana-Environment and some changes in the Neural-Network-Structure


