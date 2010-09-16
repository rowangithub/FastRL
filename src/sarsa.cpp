/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "utils.h"
#include "sarsa.h"

double & SarsaAgent::qvalue(const State & state, const int & action)
{
	return qtable_(state, action);
}

void SarsaAgent::learn(const State & state, int action, double reward, const State & post_state, int post_action)
{
	double & u = qtable_(state, action);
	const double & v = qtable_(post_state, post_action);

	u += alpha * (reward + gamma * v - u);
}

void SarsaAgent::fail(const State & state, int action, double reward)
{
	qtable_(state, action) = reward;
}
