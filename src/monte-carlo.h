/*
 * monte-carlo.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef MONTE_CARLO_H_
#define MONTE_CARLO_H_

#include <list>

#include "qlearning.h"

class VisitTable: public StateActionPairTable<u_int64_t> {
};

class MonteCarloAgent: public QLearningAgent {
public:
	MonteCarloAgent(const bool test): QLearningAgent(test) {
		visits_.load("visits.txt");
	}

	virtual ~MonteCarloAgent() {
		if (!test()) {
			visits_.save("visits.txt");
		}
	}

	virtual void learn(const State & pre_state, int pre_action, double reward, const State & state);
	virtual void fail(const State & state, int action, double reward);

private:
	std::list<std::pair<state_action_pair_t, double> > rewards_;
	VisitTable visits_;
};

#endif /* MONTE_CARLO_H_ */
