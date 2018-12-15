### Solution for Deep Reinforcement Learning Nanodegree on Udacity - Project#3: "Collaboration and Competition"
This is my solution to the second project in the "Udacity Deep Reinforcement Learning Nanodegree"

#### Project Details
This Project contains a unity-environment in which 2 Agents try to learn to play tennis

The Environment has a State-Space of 24 dimensions.

There are 2 continuous Actions, corresponding the movement of the agent

The Environment is considered as solved, if the Agents can reach an average score of 0.5+ in 100 consecutive Episodes


#### Getting Started
To run the main Notebook "Tennis.ipynb", please install:

1. pytorch 0.4
-- pip install torch

2. unityagents (you don't have to install unity3d to run the project!)
-- pip install unityagents

3. matplotlib
-- pip install matplotlib

#### Instructions
Follow the instructions inside the Jupyter-Notebook "Tennis.ipynb"

#### Credit
This Solution is based upon the implementation of a DDPG(Deep Deterministic Policy Gradient)
by the udacity Team: https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
and basically a copy of their code, adjusted to work with the Banana-Environment and some changes in the Neural-Network-Structure


