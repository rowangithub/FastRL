/*
 * pole.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef POLE_H_
#define POLE_H_

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "utils.h"

class Logger;

class Pole {
public:
	Pole() {
		reset();
	}

	const double & theta() const {	return theta_; }

	bool fail() const {
		return fabs(x_) > 1.0 || fabs(theta_) > 10.0 * one_degree;
	}

	void perturbation() { //微小扰动 - 模拟人放置杆子
		dtheta_ = irand(-one_degree, one_degree);
	}

	void reset() {
		x_ = 0.0;
		dx_ = 0.0;
		theta_ = 0.0;
		dtheta_ = 0.0;
	}

	void step(int action);

	/**
	 * state signal
	 */
	template<class State>
	State get_signal() {
		int a = (x_< 0? -1: 1) * (exp(fabs(x_)) - 1);
		int b = dx_ / 0.1;
		int c = theta_ / one_degree;
		int d = 0.1 * dtheta_ / one_degree;

		return State(a, b, c, d);
	}

	void print_state(int step) {
		std::cout << "Step " << step << ": " <<
				x_ << " " << dx_ << " " << theta_ << " " << dtheta_;
	}

	void log(Logger *logger, int action);

private:
	double x_;
	double dx_;
	double theta_;
	double dtheta_;
};

#endif /* POLE_H_ */
