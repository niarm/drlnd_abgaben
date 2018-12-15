
# coding: utf-8

# # Collaboration and Competition
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).



from unityagents import UnityEnvironment
import numpy as np



env = UnityEnvironment(file_name="env/Tennis_Linux_NoVis/Tennis.x86_64")


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


import torch
from collections import deque
from ddpg_agent import Agent
import matplotlib.pyplot as plt


# In[7]:

print("STARTING TENNIS TRAINING")
print("-------------------------------------")
def ddpg_dual(n_episodes=5000, max_t=2000, solved_at=0.5):
    
    sharedActor = Agent(state_size=state_size, action_size=action_size, random_seed=2)
    
    avg_score = []
    scores_deque = deque(maxlen=100)
    
    best_score = 0.0
    env_solved = False
    
    for i_episode in range(1, n_episodes+1):
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations                  # get the current state (for each agent)
        
        sharedActor.reset()
        
        for t in range(max_t):
            actions = sharedActor.act(states)
            
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                sharedActor.step(state, action, reward, next_state, done, t) 
            
            states  = next_states
            scores += rewards                                  # update the score (for each agent)

            if np.any(dones):                                  # exit loop if episode finished
                break
        
        score = np.max(scores)        
        avg_score.append(score)
        scores_deque.append(score)
        
        print('\rEpisode:{} \tScore:{:.3f} \tAverage Score: {:.3f} solved:{}'.format(i_episode, score, np.mean(scores_deque), env_solved), end="")
        if i_episode%10 == 0:
            print("\n")
        
        
        if score > best_score and np.mean(scores_deque) >= solved_at:
            if env_solved==False:
                env_solved = True
                print('\nEnv solved in {:.3f} episodes!\tAverage Score ={:.3f} over last {} Episodes'.format(i_episode-100, np.mean(scores_deque), 100))
            torch.save(sharedActor.actor_local.state_dict(), "actor.pth")
            torch.save(sharedActor.critic_local.state_dict(), "critic.pth")
            best_score = score
            break
        
    return avg_score

scores = ddpg_dual()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
fig.savefig('scores.png')

