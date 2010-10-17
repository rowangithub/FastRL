/*
 * tdagent.h
 *
 *  Created on: Sep 18, 2010
 *      Author: baj
 */

#ifndef TDAGENT_H_
#define TDAGENT_H_

#include <string>

#include "agent.h"
#include "table.h"

/**
 * temporal difference based method
 */
class TemporalDifferenceAgent: public Agent {
public:
	static const double alpha = 0.15; //learning rate - which is somehow good according to empirical resutls
	static const double gamma = 1.0 - 1.0e-6; //constant step parameter

public:
	TemporalDifferenceAgent(const std::string name, const PolicyType policy_type, const bool test): Agent(policy_type, test), name_(name) {
		qtable_.load(name_ + "_" + policy_name());
	}

	virtual ~TemporalDifferenceAgent() {
		if (!test()) {
			qtable_.save(name_ + "_" + policy_name());
		}
	}

	double & qvalue(const State &, const int &);

private:
	StateActionPairTable<double> qtable_;
	const std::string name_;
};

#endif /* TDAGENT_H_ */
