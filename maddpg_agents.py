import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F

from ddpg_agent import Agent
from memory import ReplayBuffer
from hyperparameters import BUFFER_SIZE, CLIP_CRITIC_GRADIENT, GAMMA
from hyperparameters import BATCH_SIZE, MULTIPLE_LEARN_PER_UPDATE, TAU
from hyperparameters import UPDATE_EVERY_N_EPISODE
from utilities import encode, decode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Maddpg():
    '''MADDPG Agent: Interacts with and learns from the environment'''

    def __init__(self, state_size, action_size, num_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # create multiple agent
        self.agents = [Agent(state_size, action_size, random_seed, num_agents)
                        for i in range(num_agents)]

        # create memory replay buffer (shared between agents)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)


    def reset(self):
        '''reset agents'''
        for agent in self.agents:
            agent.reset()

    def act(self, states, noise):
        '''
        Return action to perform for each agents (per policy)
        '''
        return [agent.act(state, noise) for agent, state in zip(self.agents, states)]

    def step(self, states, actions, rewards, next_states, dones, num_current_episode):
        '''
        save experience in replay memory, and use random sample from buffer to learn
        '''
        self.memory.add(encode(states),
                        encode(actions),
                        rewards,
                        encode(next_states),
                        dones)

        # if enough samples in the replay memory and if it is time to update
        if (len(self.memory) > BATCH_SIZE) and (num_current_episode % UPDATE_EVERY_N_EPISODE==0):

            # Note: the code only suitable for 2 agents
            assert(len(self.agents)==2)
            
            # allow to learn several times in a rwo in the same episode
            for i in range(MULTIPLE_LEARN_PER_UPDATE):
                # sample a batch of experience from the replay buffer
                experiences = self.memory.sample()
                # update #0 agent
                self.maddpg_learn(experiences, own_idx=0, other_idx=1)
                # sample another batch of experience from the replay buffer
                experiences = self.memory.sample()
                # update #1 agent
                self.maddpg_learn(experiences, own_idx=1, other_idx=0)


    def maddpg_learn(self, experiences, own_idx, other_idx, gamma=GAMMA):
        states, actions, rewards, next_states, dones = experiences
        # extract the agent's own states, actions and next_states batch
        own_states = decode(self.state_size, self.num_agents, own_idx, states)
        own_actions = decode(self.action_size, self.num_agents, own_idx, actions)
        own_next_states = decode(self.state_size, self.num_agents, own_idx, next_states)
        # extract the other agent states, actions
        other_states = decode(self.state_size, self.num_agents, other_idx, states)
        other_actions = decode(self.action_size, self.num_agents, other_idx, actions)
        other_next_states = decode(self.state_size, self.num_agents, other_idx, next_states)
        # concatenate both agent information (own agent first, other agent in second position)
        all_states = torch.cat((own_states, other_states), dim=1).to(device)
        all_actions = torch.cat((own_actions, other_actions), dim=1).to(device)
        all_next_states = torch.cat((own_next_states, other_next_states), dim=1).to(device)

        agent = self.agents[own_idx]

        # update critic
        # get predicted next-state actions and Q values from target models
        all_next_actions = torch.cat((agent.actor_target(own_next_states), agent.actor_target(other_next_states)), 
                                    dim=1).to(device)
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if (CLIP_CRITIC_GRADIENT):
            torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # update actor
        # compute actor loss
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()),
                                    dim = 1).to(device)
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # update target networks
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)

    def checkpoints(self):
        '''save checkpoints for all agents'''
        for idx, agent in enumerate(self.agents):
            actor_local_filename = 'model_dir/checkpoint_actor_local_' + str(idx) + '.pth'
            critic_local_filename = 'model_dir/checkpoint_critic_local_' + str(idx) + '.pth'
            actor_target_filename = 'model_dir/checkpoint_actor_target_' + str(idx) + '.pth'
            critic_target_filename = 'model_dir/checkpoint_critic_target_' + str(idx) + '.pth'
            torch.save(agent.actor_local.state_dict(), actor_local_filename)
            torch.save(agent.critic_local.state_dict(), critic_local_filename)
            torch.save(agent.actor_target.state_dict(), actor_target_filename)
            torch.save(agent.critic_target.state_dict(), critic_target_filename)