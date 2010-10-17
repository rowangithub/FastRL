/*
 * qlearning.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef QLEARNING_H_
#define QLEARNING_H_

#include "td-agent.h"

/**
 * off-policy Q-learning method
 */
class QLearningAgent: public TemporalDifferenceAgent {
public:
	QLearningAgent(const PolicyType policy_type, const bool test): TemporalDifferenceAgent("qlearning", policy_type, test) {

	}

	virtual ~QLearningAgent() {

	}

	virtual void learn(const State & pre_state, int pre_action, double reward, const State & state, int);
	virtual void fail(const State & state, int action, double reward);

private:
	int greedy(const State & state);
};


#endif /* QLEARNING_H_ */
