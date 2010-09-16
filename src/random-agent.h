/*
 * random-agent.h
 *
 *  Created on: Sep 16, 2010
 *      Author: baj
 */

#ifndef RANDOM_AGENT_H_
#define RANDOM_AGENT_H_

#include <cstdlib>

#include "agent.h"
#include "utils.h"

class RandomAgent: public Agent {
public:
	RandomAgent(const bool test): Agent(test) {

	}

	virtual ~RandomAgent() { }

	virtual int plan(const State &) {
		double p = prob();

		if (p < 1 / 3.0) {
			return -1;
		}
		else if (p > 2 / 3.0) {
			return 1;
		}
		else {
			return 0;
		}
	}

	virtual void learn(const State &, int, double, const State &) { }
	virtual void fail(const State &, int, double) { }
};


#endif /* RANDOM_AGENT_H_ */
