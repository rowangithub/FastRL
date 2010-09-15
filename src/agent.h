/*
 * agent.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef AGENT_H_
#define AGENT_H_

#include "state.h"

class Agent {
public:
	virtual int plan(const State &) {
		return 0;
	}

	virtual double learn(const State &, int, double, const State &) {
		return 0.0;
	}

	virtual void fail(const State &, int) {

	}
};

class RandomAgent: public Agent {
public:
	virtual int plan(const State &) {
		double prob = drand48();

		if (prob < 1 / 3.0) {
			return -1;
		}
		else if (prob > 2 / 3.0) {
			return 1;
		}
		else {
			return 0;
		}
	}
};

#endif /* AGENT_H_ */
