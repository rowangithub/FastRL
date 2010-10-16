/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include <set>

#include "utils.h"
#include "sarsa-lambda.h"

void SarsaLambdaAgent::learn(const State & state, int action, double reward, const State & post_state, int post_action)
{
	backup(state, action, reward, qvalue(post_state, post_action));
}

void SarsaLambdaAgent::backup(const State & state, int action, double reward, double post_value)
{
	const double delta = reward + gamma * post_value - qvalue(state, action);

	eligibility_[boost::tuples::make_tuple(state, action)] = 1.0;

	std::set<state_action_pair_t> zeros;
	for (std::map<state_action_pair_t, double>::iterator it = eligibility_.begin(); it != eligibility_.end(); ++it) {
		const State & state = it->first.get<0>();
		int action = it->first.get<1>();
		double & e = it->second;

		qvalue(state, action) += alpha * delta * e;
		e *= gamma * lambda;

		if (e < FLOAT_EPS) {
			zeros.insert(it->first);
		}
	}

	for (std::set<state_action_pair_t>::iterator it = zeros.begin(); it != zeros.end(); ++it) {
		eligibility_.erase(*it);
	}
}

void SarsaLambdaAgent::end()
{
	eligibility_.clear();
}
