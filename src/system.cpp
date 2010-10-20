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

double System::simulate(Agent & agent, bool verbose, Logger *logger)
{
	const int max_steps = 32768;

	int step = 0;
	double rewards = 0.0;

	State state = pole_.get_signal<State>();
	int action = agent.plan(state);

	do {
		step += 1;

		if (verbose) {
			pole_.print_state(step);
			cout << " | State: " << pole_.get_signal<State>() << " | Action: " << action;
		}

		if (logger) {
			pole_.log(logger, action);
			logger->Flush();
		}

		pole_.step(action); //taking action

		if (pole_.fail()) {
			agent.fail(state, action, get_failure_reward()); //failure state - Çø±ðÊ§°Ü×´Ì¬¸úÒ»°ãÎ´Öª×´Ì¬£¨Î´Öª×´Ì¬³õÊ¼»¯ÎªÁã£©
			if (verbose) {
				cout << " | Failure" << endl;
			}
			step += 1;
			break;
		}

		State post_state = pole_.get_signal<State>(); //observing s'
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

	if (verbose) {
		pole_.print_state(step);
		cout << " | State: " << pole_.get_signal<State>() <<  " | The End" << endl;
	}

	if (logger) {
		pole_.log(logger, 0);
		logger->Flush();
	}

	return rewards;
}
