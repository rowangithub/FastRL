/*
 * qtable.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef QTABLE_H_
#define QTABLE_H_

#include <vector>
#include <map>

#include "state.h"

class QTable {
	typedef boost::tuples::tuple<State, int> state_action_pair_t;

public:
	double & operator()(const State & state, int action) {
		return table_[boost::tuples::make_tuple(state, action)];
	}

	void save(const char *file_name) const {
		std::ofstream fout(file_name);

		if (fout.good()) {
			for (std::map<state_action_pair_t, double>::const_iterator it = table_.begin(); it != table_.end(); ++it) {
				fout << it->first << " " << std::setprecision(13) << it->second << std::endl;
			}
		}

		fout.close();
	}

	void load(const char *file_name) {
		std::ifstream fin(file_name);

		if (fin.good()) {
			state_action_pair_t state_action_pair;
			double qvalue;

			while (!fin.eof()) {
				fin >> state_action_pair >> qvalue;
				table_[state_action_pair] = qvalue;
			}
		}

		fin.close();
	}

private:
	std::map<state_action_pair_t, double> table_;
};

#endif /* QTABLE_H_ */
