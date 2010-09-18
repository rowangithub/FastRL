/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include <cassert>
#include <iostream>
#include <set>

#include "utils.h"
#include "sarsa-lambda.h"

SarsaLambdaAgent::SarsaLambdaAgent(const bool test): TemporalDifferenceAgent(test)
{
	qtable_.load("sarsa-lambda.txt");
}

SarsaLambdaAgent::~SarsaLambdaAgent()
{
	if (!test()) {
		qtable_.save("sarsa-lambda.txt");
	}
}

double & SarsaLambdaAgent::qvalue(const State & state, const int & action)
{
	return qtable_(state, action);
}

double & SarsaLambdaAgent::qvalue(const state_action_pair_t & state_action)
{
	return qtable_[state_action];
}

void SarsaLambdaAgent::learn(const State & state, int action, double reward, double bootstrap)
{
	state_action_pair_t state_action = boost::tuples::make_tuple(state, action);
	const double & u = qvalue(state, action);

	const double delta = reward + gamma * bootstrap - u;

	eligibility_trace_[state_action] = 1.0;

	std::set<state_action_pair_t> zeros;
	for (std::map<state_action_pair_t, double>::iterator it = eligibility_trace_.begin(); it != eligibility_trace_.end(); ++it) {
		double & e = it->second;
		const state_action_pair_t & sa = it->first;

		qvalue(sa) += alpha * delta * e;
		e *= gamma * lambda; //eligibility decay

		if (e < min_eligibility) {
			zeros.insert(sa);
		}
	}

	for (std::set<state_action_pair_t>::iterator it = zeros.begin(); it != zeros.end(); ++it) {
		eligibility_trace_.erase(*it);
	}
}

void SarsaLambdaAgent::learn(const State & state, int action, double reward, const State & post_state, int post_action)
{
	learn(state, action, reward, qvalue(post_state, post_action));
}

void SarsaLambdaAgent::fail(const State & state, int action, double reward)
{
	learn(state, action, reward, 0.0);

	eligibility_trace_.clear(); //prepare for new episode
}
