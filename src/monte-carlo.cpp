/*
 * monte-carlo.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "monte-carlo.h"

void MonteCarloAgent::learn(const State & state, int action, double reward, const State &)
{
	rewards_.push_back(std::make_pair(boost::tuples::make_tuple(state, action), reward));
}

/**
 * episode terminates, update the whole state-action-pair list
 * @param state
 * @param action
 * @param reward
 */
void MonteCarloAgent::fail(const State & state, int action, double reward)
{
	rewards_.push_back(std::make_pair(boost::tuples::make_tuple(state, action), reward));

	double rewards = 0.0;

	std::list<std::pair<state_action_pair_t, double> >::reverse_iterator it = rewards_.rbegin();
	for (; it != rewards_.rend(); ++it) {
		rewards += it->second;

		double & q = qtable()[it->first];
		u_int64_t & n = visits_[it->first];

		q = (q * n + rewards) / (n + 1); //average returns
		n += 1;
	}

	rewards_.clear(); //wait for new episode
}
