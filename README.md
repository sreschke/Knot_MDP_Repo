# Knot_MDP_Repo
Repository for the Knot MDP

In this class, we have implemented the Double DQN algorithm with dueling architextures and experience replay, as described in the following papers:
 
(1) Playing Atari with Deep Reinforcement Learning (2013)
by Google DeepMind authored by Hado van Hasselt et al: https://arxiv.org/pdf/1509.06461.pdf
Summary: Discusses the idea of a DQN and experience replay.
 
2) Deep Reinforcement Learning with Double Q-learning (2015)
by Google DeepMind authored by Ziyu Wang et. al.: https://arxiv.org/pdf/1511.06581.pdf
A detailed algorithm is given here: http://coach.nervanasys.com/algorithms/value_optimization/double_dqn/index.html
Introduces the idea of using TWO networks, an online network and a target network. 
This approach helps stabalize training by separating action selection from action 
evaluation. The target network's weights are frozen while the online network is updated 
at every time step. After every 1000 or so iterations, the online weights are copied to 
the target weights.
 
(3) Dueling Network Architectures for Deep Reinforcement Learning (2016)
by Google DeepMind authored by Ziyu Wang et. al.: https://arxiv.org/pdf/1511.06581.pdf
Introduces the Dueling Architexture (see figure 1 on page 1) which decouples estimates
of the state value and advantage functions.
 
(4) Prioritized Experience Replay (2016)
by Google DeepMind authored by Tom Schaul et. al: https://arxiv.org/pdf/1511.05952.pdf
Rather than uniformly sampling from the replay buffer, prioritized expereince replay 
samples important (s, a, r, s', t) transitions more frequently which leads to more 
efficient learning. We can anticipate a potential 2X speed-up.
 
 
Neural Networks will allow the algorithm to generalize learning across similar
states and thus allow us to tackle problems with larger state spaces while using
less memory. The extra bells and whistles help to stabilize and speed-up training.
