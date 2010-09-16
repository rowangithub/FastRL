/*
 * random-agent.h
 *
 *  Created on: Sep 16, 2010
 *      Author: baj
 */

#ifndef RANDOM_AGENT_H_
#define RANDOM_AGENT_H_

#include <cstdlib>

#include "agent.h"
#include "utils.h"

class RandomAgent: public Agent {
public:
	RandomAgent(const bool test): Agent(test) {

	}

	virtual ~RandomAgent() { }

	virtual int plan(const State &);
};


#endif /* RANDOM_AGENT_H_ */
