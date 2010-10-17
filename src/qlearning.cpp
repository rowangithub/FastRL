/*
 * qlearning.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "utils.h"
#include "qlearning.h"

void QLearningAgent::learn(const State & state, int action, double reward, const State & post_state, int)
{
	double & u = qvalue(state, action);
	const double & v = qvalue(post_state, greedy(post_state));

	u += alpha * (reward + gamma * v - u);
}

void QLearningAgent::fail(const State & state, int action, double reward)
{
	qvalue(state, action) = reward;
}

int QLearningAgent::greedy(const State & state)
{
	int best = -1;
    double max = qvalue(state, -1);

    for (int i = 0; i <= 1; ++i) {
        double q = qvalue(state, i);
        if (q > max) {
            max = q;
            best = i;
        }
    }

    return best;
}
