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
	double v[2];

	v[0] = qvalue(state, 0);
	v[1] = qvalue(state, 1);

	if (fabs(v[0] - v[1]) < FLOAT_EPS) {
		return RandomAgent::plan(state);
	}
	else {
		if (v[0] > v[1]) {
			return 0;
		}
		else {
			return 1;
		}
	}
}
