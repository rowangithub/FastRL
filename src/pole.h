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
		int a, b, c, d;

		if (x_ < -0.8) {
			a = -1;
		}
		else if (x_ < 0.8) {
			a = 0;
		}
		else {
			a = 1;
		}

		if (dx_ < -0.5) {
			b = -1;
		}
		else if (dx_ < 0.5) {
			b = 0;
		}
		else {
			b = 1;
		}

		if (theta_ < -6.0 * one_degree) {
			c = -3;
		}
		else if (theta_ < -one_degree) {
			c = -2;
		}
		else if (theta_ < 0) {
			c = -1;
		}
		else if (theta_ < one_degree) {
			c = 1;
		}
		else if (theta_ < 6.0 * one_degree) {
			c = 2;
		}
		else {
			c = 3;
		}

		if (dtheta_ < -50.0 * one_degree) {
			d = -1;
		}
		else if (dtheta_ < 50.0 * one_degree) {
			d = 0;
		}
		else {
			d = 1;
		}

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
