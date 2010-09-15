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
	int step = 0;
	double rewards = 0.0;
	State state = pole_.get_signal<State>();

	do {
		step += 1;
		int action = agent.plan(state);

		if (verbose) {
			pole_.print_state(step);
			cout << " | State: " << pole_.get_signal<State>() << " | Action: " << action;
		}

		if (logger) {
			pole_.log(logger, action);
			logger->Flush();
		}

		pole_.step(action);

		if (pole_.fail()) {
			agent.fail(state, action, get_terminal_reward()); //failure state - Çø±ðÊ§°Ü×´Ì¬¸úÒ»°ãÎ´Öª×´Ì¬£¨Î´Öª×´Ì¬³õÊ¼»¯ÎªÁã£©
			if (verbose) {
				cout << " | Failure" << endl;
			}
			step += 1;
			break;
		}

		State post_state = pole_.get_signal<State>();
		double reward = get_reward(action);
		agent.learn(state, action, reward, post_state);
		state = post_state;

		if (verbose) {
			cout << " | Reward: " << reward << endl;
		}

		rewards += reward;
	} while(1);

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
