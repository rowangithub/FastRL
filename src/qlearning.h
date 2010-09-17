/*
 * qlearning.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef QLEARNING_H_
#define QLEARNING_H_

#include "table.h"
#include "epsilon-agent.h"

/**
 * off-policy Q-learning method
 */
class QLearningAgent: public EpsilonAgent {
private:
	static const double alpha = 0.25;
	static const double gamma = 1.0 - 1.0e-6;

public:
	QLearningAgent(const bool test): EpsilonAgent(test) {
		qtable_.load("qlearning.txt");
	}

	virtual ~QLearningAgent() {
		if (!test()) {
			qtable_.save("qlearning.txt");
		}
	}

	double & qvalue(const State &, const int &);

	virtual void learn(const State & pre_state, int pre_action, double reward, const State & state, int);
	virtual void fail(const State & state, int action, double reward);

private:
	StateActionPairTable<double> qtable_;
};


#endif /* QLEARNING_H_ */
