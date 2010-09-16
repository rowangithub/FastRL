/*
 * epsilon-agent.h
 *
 *  Created on: Sep 16, 2010
 *      Author: baj
 */

#ifndef EPSILON_AGENT_H_
#define EPSILON_AGENT_H_

#include "random-agent.h"

class EpsilonAgent: public RandomAgent {
private:
	static const double epsilon_ = 0.1;

public:
	EpsilonAgent(const bool test): RandomAgent(test) {
	}

	virtual ~EpsilonAgent() {

	}

	virtual int plan(const State & state);
	int greedy(const State & state);

public:
	virtual double & qvalue(const State &, const int &) = 0;
};

#endif /* EPSILON_AGENT_H_ */
