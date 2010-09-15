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

class State {
public:
	State(double dx = 0.0, double theta = 0.0, double dtheta = 0.0) {
		this->dx() = dx * 100.0; //精确到厘米
		this->theta() = theta / one_degree; //精确到一度
		this->dtheta() = dtheta / one_degree; //精确到一度
	}

	int & dx() { return data_.get<0>(); }
	int & theta() {	return data_.get<1>(); }
	int & dtheta() { return data_.get<2>(); }

	const int & dx() const { return data_.get<0>(); }
	const int & theta() const {	return data_.get<1>(); }
	const int & dtheta() const { return data_.get<2>(); }

	bool operator<(const State & o) const {
		return data_ < o.data_;
	}

	bool operator==(const State & o) const {
		return data_ == o.data_;
	}

	friend std::ostream &operator<<(std::ostream & os, const State & o) {
		return os << o.data_;
	}

	friend std::istream &operator>>(std::istream & is, State & o) {
		return is >> o.data_;
	}

private:
	boost::tuples::tuple<int, int, int> data_;
};


#endif /* STATE_H_ */
