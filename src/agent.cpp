/*
 * agent.cpp
 *
 *  Created on: Oct 17, 2010
 *      Author: baj
 */

#include "agent.h"
#include "state.h"

int Agent::plan(const State & state) {
	std::vector<double> distri(3);

	for (int i = -1; i <= 1; ++i) {
		distri[i + 1] = qvalue(state, i);
	}

	return PolicyFactory::instance().CreatePolicy(policy_type_)->get_action(distri) - 1;
}
