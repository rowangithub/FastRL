/*
 * state.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef STATE_H_
#define STATE_H_

#include "utils.h"
#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"
#include "boost/tuple/tuple_io.hpp"

class State: public boost::tuples::tuple<int, int, int> {
public:
	State() { }

	State(const int & a, const int & b, const int & c): boost::tuples::tuple<int, int, int>(a, b, c) {
	}

	State operator-() const {
		return State(-get<0>(), -get<1>(), -get<2>());
	}
};


#endif /* STATE_H_ */
