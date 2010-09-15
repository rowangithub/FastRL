/*
 * qlearning.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef QLEARNING_H_
#define QLEARNING_H_

#include "agent.h"

#include <iostream>
#include<fstream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <cassert>

#include "qtable.h"

/**
 * Model-free QLearning Agent
 */
class QLearningAgent: public RandomAgent {
private:
	static const double alpha = 0.5;
	static const double gamma = 1.0 - 1.0e-6;

public:
	QLearningAgent(const double epsilon = 0.01, const bool test = false): epsilon_(epsilon), test_(test) {
		qtable_.load("qtable.txt");
	}

	~QLearningAgent() {
		if (!test_) {
			qtable_.save("qtable.txt");
		}
	}

	int plan(const State & state);
	int greedy(const State & state);
	double learn(const State & pre_state, int pre_action, double reward, const State & state);
	void fail(const State & state, int action);

private:
	QTable qtable_;
	const double epsilon_;
	const bool test_;
};


#endif /* QLEARNING_H_ */
