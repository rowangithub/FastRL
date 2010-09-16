/*
 * qtable.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef QTABLE_H_
#define QTABLE_H_

#include "state.h"

#include <fstream>
#include <iomanip>
#include <vector>
#include <map>

template<class KeyType, class DataType>
class Table {
public:
	void save(const char *file_name) const {
		std::ofstream fout(file_name);

		if (fout.good()) {
			for (typename std::map<KeyType, DataType>::const_iterator it = table_.begin(); it != table_.end(); ++it) {
				fout << std::setprecision(13) << it->first << " " << it->second << std::endl;
			}
		}

		fout.close();
	}

	void load(const char *file_name) {
		std::ifstream fin(file_name);

		if (fin.good()) {
			KeyType key;
			DataType data;

			while (!fin.eof()) {
				fin >> key >> data;
				table_[key] = data;
			}
		}

		fin.close();
	}

	std::map<KeyType, DataType> & table() { return table_; }

private:
	std::map<KeyType, DataType> table_;
};

typedef boost::tuples::tuple<State, int> state_action_pair_t;

template<class DataType>
class StateActionPairTable: public Table<state_action_pair_t, DataType> {
public:
	DataType & operator[](const state_action_pair_t & state_action_pair) {
		return this->table()[state_action_pair];
	}

	DataType & operator()(const State & state, int action) {
		return this->table()[boost::tuples::make_tuple(state, action)];
	}
};

#endif /* QTABLE_H_ */
