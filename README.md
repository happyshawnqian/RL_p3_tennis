# Project Details
In this project, two agents bounce a ball over a net. If one agent bounces the ball over the net, it receives a reward of +0.1. If the ball is hit to the ground or out of bounds, it gets a reward of -0.01. Thus, each agents tries to keep the ball in play so as to get a high reward.

The observation space is comprised of 8 variables corresponding to the position and velocity of the ball and racket. In the program, three observations are stacked to form one state. Two continuous  actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic. In order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

# Getting started
Instructions for installing the unity environment and dependencies can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet). Also, the instruction regarding setting up Python Environment can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).

# Instructions
Run the jupyter notebook file `Tennis_Work.ipynb` in the repository for training the agents and watch the result. Saved model weights can be found in the `model_dir` folder.