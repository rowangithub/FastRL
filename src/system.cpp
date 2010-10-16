/*
 * system.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include <iostream>

#include "state.h"
#include "system.h"
#include "agent.h"
#include "logger.h"

using namespace std;

double System::simulate(Agent & agent, int max_steps, bool verbose, Logger *logger)
{
	int step = 1;
	double rewards = 0.0;

	State state = pole_.get_signal<State>(step);
	int action = agent.plan(state);

	do {
		if (verbose) {
			pole_.print_state(step);
			cout << " | State: " << pole_.get_signal<State>(step) << " | Action: " << action;
		}

		if (logger) {
			pole_.log(logger, action);
			logger->Flush();
		}

		step += 1; pole_.step(action); //taking action

		State post_state = pole_.get_signal<State>(step); //observing s'
		double reward = get_reward(); //observing reward
		int post_action = agent.plan(post_state); //choosing a'

		agent.learn(state, action, reward, post_state, post_action); //learning from experience

		state = post_state;
		action = post_action;

		if (verbose) {
			cout << " | Reward: " << reward << endl;
		}

		rewards += reward;
	} while(step < max_steps);

	agent.end();

	if (verbose) {
		pole_.print_state(step);
		cout << " | State: " << pole_.get_signal<State>(step) <<  " | The End" << endl;
	}

	if (logger) {
		pole_.log(logger, 0);
		logger->Flush();
	}

	return rewards;
}
