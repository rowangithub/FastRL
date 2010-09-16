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

void QLearningAgent::learn(const State & state, int action, double reward, const State & post_state, int post_action)
{
	post_action = greedy(post_action); //use greedy-policy action

	double & u = qtable_(state, action);
	const double & v = qtable_(post_state, post_action);

	u += alpha * (reward + gamma * v - u);
}

void QLearningAgent::fail(const State & state, int action, double reward)
{
	qtable_(state, action) = reward;
}
