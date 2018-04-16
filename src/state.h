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

class State: public boost::tuples::tuple<int, int, int, int> {
public:
	State() { }

	State(const int & a, const int & b, const int & c, const int & d): boost::tuples::tuple<int, int, int, int>(a, b, c, d) {
	}

	State operator-() const {
		return State(-get<0>(), -get<1>(), -get<2>(), -get<3>());
	}

	std::vector<float> vec() const{
		//return {x,vx,ax,t,vt,at};
		return {(float)get<0>(), (float)get<1>(), (float)get<2>(), (float)get<3>()};
	}
};


#endif /* STATE_H_ */
