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
#include "Game.h"
#include "boost/tuple/tuple.hpp"

class Logger;

class Pole : public Game {
	static constexpr double time_step = 0.1;          /* seconds between state updates */

public:
	Pole() {
		reset();
	}

	bool terminate() const {
		return false; // The goal is that the pole is always on.
	}

	bool fail() const {
		return fabs(x_) > 1.0 || fabs(theta_) > 15.0 * one_degree;
	}

	double measureFailure() const {
		return -1 * std::abs(x_ - 1.0) * std::abs(theta_ - 15.0 * one_degree);
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

	int coarse_coding_x(double x) {
		return (x < 0.0? -1.0: 1.0) * (exp(fabs(x * 1.5)) - 1);
	}

	int coarse_coding_theta(double t) {
		return t / one_degree;
	}

	/**
	 * state signal
	 */
	template<class State>
	State get_signal() {
		int a = coarse_coding_x(x_);
		int b = dx_ / 0.05;
		int c = coarse_coding_theta(theta_);
		int d = coarse_coding_theta(dtheta_);

		return State(a, b, c, d);
	}

	void print_state(int step) {
		std::cout << "Step " << step << ": " <<
				x_ << " " << dx_ << " " << theta_ << " " << dtheta_;
	}

	void log(Logger *logger, int action);

    // Get and Set for simulator to access internal Pole state
	boost::tuples::tuple<double, double, double, double> getState () {
		return boost::make_tuple(x_, dx_, theta_, dtheta_);          
	}

    void setState (boost::tuples::tuple<double, double, double, double> st) {
		x_ = st.get<0>();
		dx_ = st.get<1>();
		theta_ = st.get<2>();
		dtheta_ = st.get<3>(); 
    }

    std::vector<double> getGameState () {
    	std::vector<double> data;
    	data.push_back(x_);
    	data.push_back(dx_);
    	data.push_back(theta_);
    	data.push_back(dtheta_);
		return data;          
	}

    void setGameState (std::vector<double> st) {
		x_ = st[0];
		dx_ = st[1];
		theta_ = st[2];
		dtheta_ = st[3]; 
    }

    void printGameState() {
		std::cout << "(" << x_ << ", " << dx_ << ", " << theta_ << ", " << dtheta_ << ")\n";
	}

    int actions() {return 3;}
    int inputs() {return 4;}

private:
	double x_;
	double dx_;
	double theta_;
	double dtheta_;
};

class UCTPoleSimulator : public UCTGameSimulator {
public:
	// Game engine
    Pole pole_;

    UCTPoleSimulator () {
		init (&pole_);
	}

	UCTPoleSimulator (net::ComputationGraph* policy) {
		init (&pole_, policy);
	}
};

#endif /* POLE_H_ */
