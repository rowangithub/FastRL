/*
 * qtable.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef QTABLE_H_
#define QTABLE_H_

#include "state.h"
#include "auto-table.h"

typedef boost::tuples::tuple<State, int> state_action_pair_t;

template<class DataType>
class StateActionPairTable: public AutoTable<state_action_pair_t, DataType> {
public:
	DataType & operator[](const state_action_pair_t & state_action_pair) {
		return this->table()[state_action_pair];
	}

	DataType & operator()(const State & state, int action) {
		return this->table()[boost::tuples::make_tuple(state, action)];
	}
};

class QTable: public StateActionPairTable<double> {
};

#endif /* QTABLE_H_ */
