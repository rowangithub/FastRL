/*
 * qlearning.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef SARSA_H_
#define SARSA_H_

#include "td-agent.h"

/**
 * on-policy Sarsa method
 */
class SarsaAgent: public TemporalDifferenceAgent {
public:
	SarsaAgent(const PolicyType policy_type, const bool test): TemporalDifferenceAgent("sarsa", policy_type, test) {

	}

	virtual ~SarsaAgent() {

	}

	virtual void learn(const State & pre_state, int pre_action, double reward, const State & state, int);
	virtual void fail(const State & state, int action, double reward);
};


#endif /* QLEARNING_H_ */
