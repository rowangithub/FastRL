/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "utils.h"
#include "qlearning.h"

int QLearningAgent::plan(const State & state)
{
	if (drand48() < epsilon_) {
		return RandomAgent::plan(state);
	}
	else {
		return this->greedy(state);
	}
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

double QLearningAgent::learn(const State & pre_state, int pre_action, double reward, const State & state)
{
	int action = greedy(state);
	double & u = qtable_(pre_state, pre_action);
	double v = qtable_(state, action);

	double error = alpha * (reward + gamma * v - u);

	u = u + error;

	return error;
}

void QLearningAgent::fail(const State & state, int action)
{
	qtable_(state, action) = -10.0;
}
