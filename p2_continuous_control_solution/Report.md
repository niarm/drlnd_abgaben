### Report for my Solution on: Deep Reinforcement Learning Nanodegree on Udacity - Project#2: "Continuous Control"

#### 1. Environment
The Given Environment "Reacher_multi.app" has the following structure:

States: 33 float numbers
Actions: 4 contiuous actions (values between -1 and 1)


#### 2. Learning Algorithm
The used Algorithm is a Deep-Deterministic-Policy-Gradient (DDGP)
Multiple Agents where used (The MultiAgent-Environment has 20 Agents)
The Agents pushed their observations into a shared Replay Buffer, what dramatically increases the exploration ability of the Algorithm.
Every 20 Time-Steps, the Agents take 10 learning steps. 

#### 3. Hyperparameters
The Agents was trained with the following Hyperparameters:

Number of Episodes: 300
Maximum Steps per Episode: 600


#### 4. Neural Network
The Neural Network for DDPG-Actor was:

Linear (Fully-Connected Layers) with units:
[256, 4]

The Neural Network for the DDPG-Critic was:
[256, 256,128,1]



#### 5. Results
After 300 Episodes, the Agents scored 73.13 points on Average over a time-window of 100 Episodes

Episode 100     Average Score: 14.26
Episode 200	    Average Score: 43.33
Episode 300	    Average Score: 73.13
