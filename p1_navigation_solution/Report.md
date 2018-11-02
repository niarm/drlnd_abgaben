### Report for my Solution on: Deep Reinforcement Learning Nanodegree on Udacity - Project#1: "Navigation"

#### 1. Environment
The Given Environment "Banana.app" has the following structure:

States: 37 float numbers
Actions: 4 discrete actions


#### 2. Learning Algorithm
The used Algorithm is a Deep-Q-Network (DQN)

In DQN, an Agent, interacting with the Environment is using a Neural Network as an estimator of the Q-Function.
Q: S x A -> R

In Reinforcement-learning, a Q-Function is a Function that "rates" a given state and an action that the Agent choses.
Classical Reinforcement algorithms use a so called Q-table as Q-Function. This refers to a table in which each combination of possible states and actions has a value that describes, how "good" it is for the agent to be in that state and take the corresponding action. A so called "policy" is then used to chose the best action for the current state.  

Environments with non-discrete states (such as our Banana-3d-world in this project) make it hard of course to be stored in a table. One can take approaches, where the contiuous world is discretized of course such as to lay some kind of discrete grid over the continuous environment, but Deep-Q-Learning takes a different approach, that produces much better outcomes:
Instead of using a grid, DQN uses a deep neural network to estimate the values for each state and action.
The neural Networks takes the state-space of the environment as input (37 Raycasts in the Banana-environment) and outputs a distribution over all possible actions (4 in our the Banana-Environment).
We then chose the action to take by a "epsilon-greedy" policy.

"Epsilon-Greedy" choses the maximal value in our output except for a random probability "epsilon", where it choses a random action.

A good and more complete description of the algorithm can be found here:
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf


###### 2.1 Experience Replay
There are 2 major discoveries made by the team that invented DQN.
The first is a so called "Experience Replay".
If the Agent is interacting with the environment, directly learning from the observation it makes, all the "experience" the agent makes is highly correlated. That makes it practically impossible for the agent to generalize.
This problem is tackled by the idea of "experience replay" where a "replay buffer" is introduced.
The agent no longer directly learns from his observations but stores all his observations in a buffer.
After some steps, the agent then trains the Neural Network by randomly sampling batches out of that buffer.
This helps to generalize and avoid correlations during training.

###### 2.2 Fixed Q-targets
The second problem one faces in training an Agent with DQN is, that inside a neural network, "small changes in the network can result in big changes of the policy. The solution is somehow simular to what the replay buffer does:
Instead of changing the networks weights after each step, in addition to the "local network", a second network is introduced which we call "target network". This target-network is only updated periodically.
The local network now adjusts in direction towards the values of the target values, reducing correlations.



#### 3. Hyperparameters
The Agent was trained with the following Hyperparameters:

Number of Episodes: 2000
Maximum Steps per Episode: 1000
epsilon_start:1.0
epsilon_decay: 0.995 (decrease of epsilon after each episode) 
epsilon_end: 0.01 (minimal epsilon)


#### 4. Neural Network
The Neural Network for Q-Value estimation was structured:

Linear (Fully-Connected Layers) with units:
[256, 128, 128, 32]

Each Layer was activated with a Relu
After each activation during learning, a Dropout was applied with p=0.15


#### 5. Results
The best Agent scored 15.78 points in average on 100 Episodes


###### 5.1 Result Plot
Episode 100	    Average Score: 0.10 - epsilon: 0.6057704364907278
Episode 200	    Average Score: 2.79 - epsilon: 0.36695782172616715
Episode 300	    Average Score: 8.90 - epsilon: 0.22229219984074702
Episode 400	    Average Score: 9.30 - epsilon: 0.13465804292601349
Episode 500	    Average Score: 12.05 - epsilon: 0.08157186144027828
Episode 600	    Average Score: 13.04 - epsilon: 0.049413822110038545
Episode 700	    Average Score: 14.61 - epsilon: 0.029933432588273214
Episode 800	    Average Score: 15.48 - epsilon: 0.018132788524664028
Episode 900	    Average Score: 15.71 - epsilon: 0.010984307219379798
Episode 1000	Average Score: 15.52 - epsilon: 0.01036634861955105
Episode 1100	Average Score: 15.07 - epsilon: 0.01
Episode 1200	Average Score: 15.78 - epsilon: 0.01
Episode 1300	Average Score: 14.96 - epsilon: 0.01
Episode 1400	Average Score: 13.57 - epsilon: 0.01
Episode 1500	Average Score: 13.63 - epsilon: 0.01
Episode 1600	Average Score: 14.49 - epsilon: 0.01
Episode 1700	Average Score: 14.92 - epsilon: 0.01
Episode 1800	Average Score: 14.85 - epsilon: 0.01
Episode 1900	Average Score: 14.39 - epsilon: 0.01
Episode 2000	Average Score: 14.57 - epsilon: 0.01

Please see "main_solution.ipynb" for a visualized graph of the learning curve


#### 6. Future Improvements
The Network coul probably be improved by changing the algorithm for the Replay-Buffer.
Instead of taking random batches, a prioritized Replay Buffer would probably improve the results.