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

class Pole {
public:
	Pole() {
		reset();
	}

	const double & x() const { return x_; }
	const double & dx() const { return dx_; }
	const double & theta() const {	return theta_; }
	const double & dtheta() const { return dtheta_; }

	bool fail() const {
		return fabs(theta_) > 10.0 * one_degree || fabs(dx_) > 1.0;
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
		return State(dx_, theta_, dtheta_);
	}

	void print_state(int step) {
		std::cout << "Step " << step << ": " <<
				x_ << " " << dx_ << " " << theta_ << " " << dtheta_;
	}

	template<class Logger>
	void log(Logger *logger, int action);

private:
	double x_;
	double dx_;
	double theta_;
	double dtheta_;
};

#endif /* POLE_H_ */
