/*
 * system.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <cstdlib>
#include <cmath>

#include "pole.h"

class Agent;
class Logger;

class System {
public:
	System() {
		pole_.perturbation();
	}

	void reset() {
		pole_.reset();
	}

	double get_reward() {
		return 1.0;
	}

	double get_failure_reward() {
		return -1.0;
	}

	double simulate(Agent & agent, bool verbose = true, Logger *logger = 0);

private:
	Pole pole_;
};

#endif /* SYSTEM_H_ */
