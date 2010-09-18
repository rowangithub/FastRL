/*
 * qlearning.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef SARSA_H_
#define SARSA_H_

#include "table.h"
#include "td-agent.h"

/**
 * on-policy Sarsa method
 */
class SarsaAgent: public TDAgent {
public:
	SarsaAgent(const bool test): TDAgent(test) {
		qtable_.load("sarsa.txt");
	}

	virtual ~SarsaAgent() {
		if (!test()) {
			qtable_.save("sarsa.txt");
		}
	}

	double & qvalue(const State &, const int &);

	virtual void learn(const State & pre_state, int pre_action, double reward, const State & state, int);
	virtual void fail(const State & state, int action, double reward);

private:
	StateActionPairTable<double> qtable_;
};


#endif /* QLEARNING_H_ */
