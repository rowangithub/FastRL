/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "utils.h"
#include "qlearning.h"

void QLearningAgent::learn(const State & state, int action, double reward, const State & post_state, int)
{
	if (test()) return;

	double & u = qvalue(state, action);
	const double & v = qvalue(post_state, greedy(post_state));

	u += alpha * (reward + gamma * v - u);
}

void QLearningAgent::fail(const State & state, int action, double reward)
{
	if (test()) return;

	qvalue(state, action) = reward;
}
