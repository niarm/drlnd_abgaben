
# coding: utf-8

# In[1]:


from unityagents import UnityEnvironment
import numpy as np
import torch


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


print("DEVICE:",device)


# ## Load the Environment

# First, we load the unity-Environment. There are 2 different versions of the Environment: 
# There is a Single-Agent Environment and a Multi-Agent Environment. We will use the Multi-Agent Env in this notebook.

# In[4]:


#single Agent Version:
#env = UnityEnvironment(file_name='env/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

#multiple-Agent Version
env = UnityEnvironment(file_name='env/Reacher_Linux_NoVis/Reacher.x86_64')


# In[5]:


# get the "brain" from env
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

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


# ### Train the Agents

# This section will train our Agents on the Environment. Since this is quite computational-intensive, this procedure could take some time. 

# In[6]:


from collections import deque
from ddpg_agent import Agent
import time

import matplotlib.pyplot as plt


# In[7]:


agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)


# In[8]:


def ddpg(n_episodes=1000, max_t=1000):
    mean_scores = []                               
    min_scores = [] 
    max_scores = []
    scores_deque = deque(maxlen=100)  # mean scores from most recent episodes
    
    for i_episode in range(1, n_episodes+1):
        
        env_info = env.reset(train_mode=True)[brain_name]       # reset Env
        states = env_info.vector_observations                   # get states      
        scores = np.zeros(num_agents)                           # initialize scores
        agent.reset()                                           # reset agents noise
        start_time = time.time()                                # save start time
        
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)         # take an action
            env_info = env.step(actions)[brain_name]            # send actions to the unity-environment
            next_states = env_info.vector_observations          # get next_states
            rewards = env_info.rewards                          # get rewards
            dones = env_info.local_done                         # get dones
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)             
            
            states = next_states
            scores += rewards        
            
            if np.any(dones):                                   
                break

        duration = time.time() - start_time  # calc duration
        
        min_scores.append(np.min(scores))    # save worst results
        max_scores.append(np.max(scores))    # save best results
        mean_scores.append(np.mean(scores))  # save mean scores
        scores_deque.append(np.mean(scores)) # save mean scores to scrolling score-window
                
        print('\rEpisode {} \t duration:{}s \tWorst: {:.2f}\tBest: {:.2f}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], np.mean(scores_deque)))
                  
        if np.mean(scores_deque) >= 30 and i_episode >= 100:
            print('\nEnv solved in {} episodes!\tAverage Score ={:.1f} over last {} Episodes'.format(i_episode-100, np.mean(scores_deque), 100))
            torch.save(agent.actor_local.state_dict(), "actor.pth")
            torch.save(agent.critic_local.state_dict(), "critic.pth")  
            break
            
    return mean_scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
fig.savefig('scores_figure.png')

