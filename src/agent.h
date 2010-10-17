/*
 * agent.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef AGENT_H_
#define AGENT_H_

#include "policy.h"

class State;

enum AlgorithmType {
	AT_None,

	AT_MonteCarlo,
	AT_Sarsa,
	AT_QLearning,
	AT_SarsaLambda
};

class Agent {
public:
	Agent(const PolicyType policy_type, const bool test): policy_type_(policy_type), test_(test) {

	}

	virtual ~Agent() { }

	int plan(const State & state);

	virtual double & qvalue(const State &, const int &) = 0;
	virtual void learn(const State &, int, double, const State &, int) = 0; //learning from full quintuple
	virtual void fail(const State &, int, double) = 0; //inform the agent about failure

	const bool & test() const { return test_; }

private:
	const PolicyType policy_type_;
	const bool test_;
};

#endif /* AGENT_H_ */
