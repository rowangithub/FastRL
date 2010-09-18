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

double & SarsaLambdaAgent::eligibility(const state_action_pair_t & state_action)
{
	return eligibility_[state_action];
}

void SarsaLambdaAgent::learn(const State & state, int action, double reward, const State & post_state, int post_action)
{
	state_action_pair_t state_action = boost::tuples::make_tuple(state, action);

	const double & u = qvalue(state, action);
	const double & v = qvalue(post_state, post_action);

	const double delta = reward + gamma * v - u;
	double & e = eligibility(state_action);

	e += 1.0;

	std::set<state_action_pair_t> zeros;
	for (std::map<state_action_pair_t, double>::iterator it = eligibility_.begin(); it != eligibility_.end(); ++it) {
		double & e = it->second;
		const state_action_pair_t & sa = it->first;

		if (sa.get<0>() == state && sa.get<1>() != action) { //set to be zero
			e = 0.0;
		}

		if (e > min_eligibility) {
			qvalue(sa) += alpha * delta * e;
			e *= gamma * lambda; //normal decay
		}
		else {
			zeros.insert(sa);
		}
	}

	for (std::set<state_action_pair_t>::iterator it = zeros.begin(); it != zeros.end(); ++it) {
		eligibility_.erase(*it);
	}
}

void SarsaLambdaAgent::fail(const State & state, int action, double reward)
{
	qvalue(state, action) = reward;
	eligibility_.clear(); //prepare for new episode
}
