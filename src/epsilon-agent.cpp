/*
 * epsilon-agent.cpp
 *
 *  Created on: Sep 16, 2010
 *      Author: baj
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "epsilon-agent.h"

/**
 * epsilon-greedy policy
 * @param state
 * @return
 */
int EpsilonAgent::plan(const State & state)
{
	if (test()) {
		return greedy(state);
	}
	else {
		if (prob() < epsilon_) {
			return RandomAgent::plan(state);
		}
		else {
			return greedy(state);
		}
	}
}

int EpsilonAgent::greedy(const State & state)
{
	std::vector<int> actions;
	double max = -10000.0;

	for (int i = -1; i <= 1; ++i) {
		double q = qvalue(state, i);
		if (q > max) {
			max = q;
			actions.clear();
			actions.push_back(i);
		}
		else if (q > max - FLOAT_EPS) {
			actions.push_back(i);
		}
	}

	if (!actions.empty()) {
		std::random_shuffle(actions.begin(), actions.end());
		return actions.front();
	}
	else {
		assert(0);
		return RandomAgent::plan(state);
	}
}
