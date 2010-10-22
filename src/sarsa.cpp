/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "utils.h"
#include "sarsa.h"

void SarsaAgent::learn(const State & state, int action, double reward, const State & post_state, int post_action)
{
	if (test()) return;

	double & u = qvalue(state, action);
	const double & v = qvalue(post_state, post_action);

	u += alpha * (reward + gamma * v - u);
}

void SarsaAgent::fail(const State & state, int action, double reward)
{
	if (test()) return;

	qvalue(state, action) = reward;
}
