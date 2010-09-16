/*
 * random-agent.cpp
 *
 *  Created on: Sep 16, 2010
 *      Author: baj
 */

#include "random-agent.h"

int RandomAgent::plan(const State &)
{
	double p = prob();

	if (p < 1 / 3.0) {
		return -1;
	}
	else if (p > 2 / 3.0) {
		return 1;
	}
	else {
		return 0;
	}
}
