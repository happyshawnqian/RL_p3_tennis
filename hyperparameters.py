import torch.nn.functional as F

# Default hyperparameters

SEED = 10                           # Random seed

N_EPISODES = 10000                  # Max number of episodes
N_STEPS = 1000                      # Max number of steps per episodes
UPDATE_EVERY_N_EPISODE = 4          # number of episodes between learning process
MULTIPLE_LEARN_PER_UPDATE = 3       # number of multiple learning process performed in a row

BUFFER_SIZE = int(1e5)              # replay buffer size
BATCH_SIZE = 200                    # minibatch size

ACTOR_FC1_UNITS = 400               # number of units for the layer 1 in the actor model
ACTOR_FC2_UNITS = 300               # number of units for the layer 2 in the actor model
CRITIC_FCS1_UNITS = 400             # number of units for the layer 1 in the critic model
CRITIC_FC2_UNITS = 300              # number of units for the layer 2 in the critic model
NON_LIN = F.relu                    # non linearity operator used in the model
LR_ACTOR = 1e-4                     # learning rate of the actor
LR_CRITIC = 1e-3                    # learning rate of the critic
WEIGHT_DECAY = 0                    # L2 weight decay

GAMMA = 0.995                       # discount factor
TAU = 1e-3                          # parameter for soft update
CLIP_CRITIC_GRADIENT = False        # clip gradient during critic optimization

ADD_OU_NOISE = True                 # add Ornstein-Uhlenbeck noise
MU = 0.                             # noise parameter - mean
THETA = 0.15                        # noise parameter - variance
SIGMA = 0.2                         # noise parameter - variance
NOISE = 0.0                         # initial noise amplitude
NOISE_REDUCTION = 1.0               # noise amplitude decay ratio