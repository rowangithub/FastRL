/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "utils.h"
#include "qlearning.h"

/**
 * epsilon-greedy policy
 * @param state
 * @return
 */
int QLearningAgent::plan(const State & state)
{
	if (test()) {
		return greedy(state);
	}
	else {
		if (prob() < epsilon_) {
			return RandomAgent::plan(state);
		}
		else {
			return greedy(state);
		}
	}
}

void QLearningAgent::learn(const State & state, int action, double reward, const State & post_state)
{
	double & u = qtable_(state, action);
	const double & v = qtable_(post_state, greedy(post_state));

	u += alpha * (reward + gamma * v - u);
}

void QLearningAgent::fail(const State & state, int action, double reward)
{
	qtable_(state, action) = reward;
}

int QLearningAgent::greedy(const State & state)
{
	std::vector<int> actions;
	double max = -10000.0;

	for (int i = -1; i <= 1; ++i) {
		double q = qtable_(state, i);
		if (q > max) {
			max = q;
			actions.clear();
			actions.push_back(i);
		}
		else if (q > max - FLOAT_EPS) {
			actions.push_back(i);
		}
	}

	if (!actions.empty()) {
		random_shuffle(actions.begin(), actions.end());
		return actions.front();
	}
	else {
		assert(0);
		return RandomAgent::plan(state);
	}
}
