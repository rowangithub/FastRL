/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "utils.h"
#include "qlearning.h"

double & QLearningAgent::qvalue(const State & state, const int & action)
{
	return qtable_(state, action);
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
