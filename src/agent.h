/*
 * agent.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef AGENT_H_
#define AGENT_H_

class State;

enum AgentType {
	AT_None,
	AT_MonteCarlo,
	AT_Sarsa,
	AT_QLearning,
	AT_SarsaLambda
};

class Agent {
public:
	Agent(const bool test): test_(test) {

	}

	virtual ~Agent() { }

	virtual int plan(const State &) = 0;
	virtual void learn(const State &, int, double, const State &, int) { } //learning from SARSA incremental
	virtual void end() { } //episode ends

	const bool & test() const { return test_; }

private:
	const bool test_;
};

#endif /* AGENT_H_ */
