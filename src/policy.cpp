/*
 * policy.cpp
 *
 *  Created on: Oct 17, 2010
 *      Author: baj
 */

#include <cassert>
#include <algorithm>

#include "policy.h"
#include "utils.h"

std::string Policy::name(PolicyType type)
{
	switch(type) {
	case PT_Random: return "random";
	case PT_Greedy: return "greedy";
	case PT_EpsilonGreedy: return "epsilon-greedy";
	case PT_Softmax: return "softmax";
	default: return "none";
	}
}

int RandomPolicy::get_action(std::vector<double> distri)
{
	return rand() % distri.size();
}

int GreedyPolicy::get_action(std::vector<double> distri)
{
	int best = 0;
	double max = distri[0];

	for (uint i = 1; i < distri.size(); ++i) {
		if (distri[i] > max) {
			max = distri[i];
			best = i;
		}
	}

	return best;
}

int EpsilonGreedyPolicy::get_action(std::vector<double> distri)
{
	if (get_prob() < epsilon_) {
		return rand() % distri.size();
	}
	else {
		return GreedyPolicy::get_action(distri);
	}
}

int SoftmaxPolicy::get_action(std::vector<double> distri)
{
	double sum = 0.0;
	std::vector<double> probs(distri.size());

	for (uint i = 0; i < distri.size(); ++i) {
		probs[i] = exp(distri[i]);
		sum += probs[i];
	}

	for (uint i = 0 ; i < probs.size(); ++i) {
		probs[i] /= sum;

		if (i) {
			probs[i] += probs[i-1];
		}
	}

	double prob = get_prob();
	for (uint i = 0 ; i < probs.size(); ++i) {
		if (prob < probs[i]) {
			return i;
		}
	}

	assert(0);
	return 0;
}
