/*
 * qlearning.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef SARSALAMBDA_H_
#define SARSALAMBDA_H_

#include <list>

#include "table.h"
#include "td-agent.h"

/**
 * on-policy Sarsa-Lambda method
 */
class SarsaLambdaAgent: public TemporalDifferenceAgent {
private:
	static const double lambda = 0.8;
	static const double min_eligibility = 1.0e-3;

public:
	SarsaLambdaAgent(const bool test);
	virtual ~SarsaLambdaAgent();

	double & qvalue(const State &, const int &);

	virtual void learn(const State & pre_state, int pre_action, double reward, const State & state, int);
	virtual void fail(const State & state, int action, double reward);

private:
	double & qvalue(const state_action_pair_t &);
	void backup(const State & pre_state, int pre_action, double reward, double bootstrap);

private:
	StateActionPairTable<double> qtable_;
	StateActionPairTable<double> eligibility_trace_;
};


#endif /* QLEARNING_H_ */
