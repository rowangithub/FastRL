/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include <cassert>
#include <iostream>

#include "utils.h"
#include "sarsa-lambda.h"

SarsaLambdaAgent::SarsaLambdaAgent(const bool test): TemporalDifferenceAgent(test)
{
	sarsa_lambda_.load("sarsa-lambda.txt");

	std::map<state_action_pair_t, boost::tuples::tuple<double, double> >::iterator it = sarsa_lambda_.table().begin();
	for (; it != sarsa_lambda_.table().end(); ++it) {
		if (it->second.get<1>() >= min_eligibility) {
			nonzero_state_action_.insert(it->first);
		}
	}
}

SarsaLambdaAgent::~SarsaLambdaAgent()
{
	if (!test()) {
		sarsa_lambda_.save("sarsa-lambda.txt");
	}
}

double & SarsaLambdaAgent::qvalue(const State & state, const int & action)
{
	return sarsa_lambda_(state, action).get<0>();
}

double & SarsaLambdaAgent::qvalue(const state_action_pair_t & state_action)
{
	return sarsa_lambda_[state_action].get<0>();
}

double & SarsaLambdaAgent::eligibility(const state_action_pair_t &state_action)
{
	return sarsa_lambda_[state_action].get<1>();
}

void SarsaLambdaAgent::learn(const State & state, int action, double reward, const State & post_state, int post_action)
{
	state_action_pair_t state_action = boost::tuples::make_tuple(state, action);

	const double & u = qvalue(state, action);
	const double & v = qvalue(post_state, post_action);

	const double delta = reward + gamma * v - u;
	double & e = eligibility(state_action);

	e += 1.0;

	nonzero_state_action_.insert(state_action);
	std::set<state_action_pair_t> zero_state_action_;

	for (std::set<state_action_pair_t>::iterator it = nonzero_state_action_.begin(); it != nonzero_state_action_.end(); ++it) {
		double & e = eligibility(*it);

		if (it->get<0>() == state && it->get<1>() != action) { //set to be zero
			e = 0.0;
		}

		if (e >= min_eligibility) {
			qvalue(*it) += alpha * delta * e;
			e *= gamma * lambda; //normal decay
		}
		else {
			zero_state_action_.insert(*it);
		}
	}

	for (std::set<state_action_pair_t>::iterator it = zero_state_action_.begin(); it != zero_state_action_.end(); ++it) {
		nonzero_state_action_.erase(*it);
	}
}

void SarsaLambdaAgent::fail(const State & state, int action, double reward)
{
	qvalue(state, action) = reward;
}
