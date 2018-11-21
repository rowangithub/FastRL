# pole
Reinforcement Learning algorithms for an inverted pendulum with a cart.


# Dependencies
- libboost-dev


# Compile
./trainall.sh Debug


# Train DQN
./Debug/pole train pole


# Train using MCTS and Supervised Learning
./Debug/pole cegis pole


# Test the trained neural network in 1000 episodes each of which consists of 200 steps.
./Debug/pole play pole supervised_agent.network 200 1000
