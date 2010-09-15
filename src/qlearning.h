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
#include <iomanip>
#include <algorithm>
#include <cassert>

#include "qtable.h"

/**
 * Model-free QLearning Agent
 */
class QLearningAgent: public RandomAgent {
private:
	static const double epsilon_ = 0.9;
	static const double alpha = 0.5;
	static const double gamma = 1.0 - 1.0e-6;

public:
	QLearningAgent(const bool test): RandomAgent(test) {
		qtable_.load("qtable.txt");
	}

	virtual ~QLearningAgent() {
		if (!test()) {
			qtable_.save("qtable.txt");
		}
	}

	virtual int plan(const State & state);
	virtual void learn(const State & pre_state, int pre_action, double reward, const State & state);
	virtual void fail(const State & state, int action, double reward);

	int greedy(const State & state);

public:
	QTable & qtable() { return qtable_; }

private:
	QTable qtable_;
};


#endif /* QLEARNING_H_ */
