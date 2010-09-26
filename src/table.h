/*
 * qtable.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef QTABLE_H_
#define QTABLE_H_

#include "state.h"

#include <iostream>
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
			fout << *this;
		}

		fout.close();
	}

	void load(const char *file_name) {
		std::ifstream fin(file_name);

		if (fin.good()) {
			fin >> *this;
		}

		fin.close();
	}

	friend std::ostream & operator<<(std::ostream & os, const Table & o) {
		os << o.size() << std::endl;
		for (typename std::map<KeyType, DataType>::const_iterator it = o.begin(); it != o.end(); ++it) {
			os << std::setprecision(13) << it->first << " " << it->second << std::endl;
		}

		return os;
	}

	friend std::istream & operator>>(std::istream & is, Table & o) {
		uint size;
		KeyType key;
		DataType data;

		is >> size;
		for (uint i = 0; i < size; ++i) {
			is >> key >> data;
			o[key] = data;
		}

		return is;
	}
};

typedef boost::tuples::tuple<State, int> state_action_pair_t;

template<class DataType>
class ActionDistribution: public boost::tuples::tuple<DataType, DataType, DataType> {
public:
	DataType & operator[](const int action) {
		if (action < 0) {
			return this->template get<0>();
		}
		else if (action > 0) {
			return this->template get<2>();
		}
		else {
			return this->template get<1>();
		}
	}
};

template<class DataType>
class StateActionPairTable: public Table<State, ActionDistribution<DataType> > {
public:
	DataType & operator()(const State & state, int action) {
		return this->operator[](state)[action];
	}

	DataType & operator()(const state_action_pair_t & pair) {
		return this->operator[](pair.get<0>())[pair.get<1>()];
	}
};

#endif /* QTABLE_H_ */
