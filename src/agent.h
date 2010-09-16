/*
 * agent.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef AGENT_H_
#define AGENT_H_

#include <cstdlib>

#include "utils.h"

class State;

enum AgentType {
	AT_None,
	AT_QLearning,
	AT_MonteCarlo
};

class Agent {
public:
	Agent(const bool test): test_(test) {

	}

	virtual ~Agent() { }

	virtual int plan(const State &) = 0;
	virtual void learn(const State &, int, double, const State &) = 0;
	virtual void fail(const State &, int, double) = 0;

	const bool & test() const { return test_; }

private:
	const bool test_;
};

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

#endif /* AGENT_H_ */
