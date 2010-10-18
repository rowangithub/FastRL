/*
 * random-agent.cpp
 *
 *  Created on: Sep 16, 2010
 *      Author: baj
 */

#include "random-agent.h"

int RandomAgent::plan(const State &)
{
	return rand() % 2;
}
