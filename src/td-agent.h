/*
 * tdagent.h
 *
 *  Created on: Sep 18, 2010
 *      Author: baj
 */

#ifndef TDAGENT_H_
#define TDAGENT_H_

#include "epsilon-agent.h"

/**
 * temporal difference based method
 */
class TemporalDifferenceAgent: public EpsilonAgent {
public:
	static const double alpha = 0.15; //learning rate - which is somehow good according to empirical resutls
	static const double gamma = 1.0 - 1.0e-6; //constant step parameter

public:
	TemporalDifferenceAgent(const bool test): EpsilonAgent(test) {
	}

	virtual ~TemporalDifferenceAgent() {
	}
};

#endif /* TDAGENT_H_ */
