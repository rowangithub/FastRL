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
class Table: public std::map<KeyType, DataType> {
public:
	void save(const char *file_name) const {
		std::ofstream fout(file_name);

		if (fout.good()) {
			for (typename std::map<KeyType, DataType>::const_iterator it = this->begin(); it != this->end(); ++it) {
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
				this->operator[](key) = data;
			}
		}

		fin.close();
	}
};

typedef boost::tuples::tuple<State, int> state_action_pair_t;

template<class DataType>
class StateActionPairTable: public Table<state_action_pair_t, DataType> {
public:
	DataType & operator()(const State & state, int action) {
		return this->operator[](boost::tuples::make_tuple(state, action));
	}
};

#endif /* QTABLE_H_ */
