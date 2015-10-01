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

void SarsaLambdaAgent::fail(const State & state, int action, double reward)
{
	backup(state, action, reward, 0.0);

	eligibility_.clear();
}

void SarsaLambdaAgent::backup(const State & state, int action, double reward, double post_qvalue)
{
	eligibility_[boost::tuples::make_tuple(state, action)] = 1.0;

	std::set<state_action_pair_t> zeros;
	for (std::map<state_action_pair_t, double>::iterator it = eligibility_.begin(); it != eligibility_.end(); ++it) {
		const State & state = it->first.get<0>();
		int action = it->first.get<1>();
		double & eligbility = it->second;

        const double delta = reward + gamma * post_qvalue - qvalue(state, action); //这里跟 Sutton 书上是不一样的

		qvalue(state, action) += alpha * delta * eligbility;
		eligbility *= gamma * lambda;

		if (eligbility < FLOAT_EPS) {
			zeros.insert(it->first);
		}
	}

	for (std::set<state_action_pair_t>::iterator it = zeros.begin(); it != zeros.end(); ++it) {
		eligibility_.erase(*it);
	}
}
