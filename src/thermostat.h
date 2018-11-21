#ifndef THERMOSTAT_H_
#define THERMOSTAT_H_

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "utils.h"
#include "Game.h"
#include "boost/tuple/tuple.hpp"

class Logger;

class Thermostat : public Game {
	static constexpr double time_step = 0.2;

private:
	double x;
	int mode; // mode = 0 heating, mode = 1 cooling.
	
public:
	Thermostat() {
		reset();
	}

	bool terminate() const {
		if (inv_mode)
			return (game_step >= 1 && (x >= 68 && x <= 72));
		else
			return false;
	}

	bool fail() const {
		if (inv_mode)
			return game_step >= 1 && (x < 68 || x > 72);
		else 
			return ( x <= 63 || x >= 76);
	}

	double measureFailure() const {
		return -1.0;
	}

	void perturbation() { 
		// Perturbate x from 68 - 72.
		x = ((rand() % 101 - 50) / 25.f) + 70.0;
    }

	void reset()
	{
		if (inv_mode) game_step = 0;
		x = 0;
		mode = 0;
	}

	void step( int ac )
	{
		if (inv_mode) game_step += 1;

		mode = ac;

		if (mode == 0) {
			double ddx = 40-0.5*x;
			x  += time_step * ddx;
		}

		if (mode == 1) {
			double ddx = 30-0.5*x;
			x  += time_step * ddx;
		}
	}

    std::vector<double> getGameState () {
    	std::vector<double> data;
    	data.push_back(x);
		return data;          
	}

    void setGameState (std::vector<double> st) {
		x = st[0];
    }

    void printGameState() {
		std::cout << "(" << x << ")\n";
	}

    int actions() {return 2;}
    int inputs() {return 1;}
};

class UCTThermostatSimulator : public UCTGameSimulator {
public:
	// Game engine
    Thermostat t_;

    UCTThermostatSimulator () {
		init (&t_);
	}

	UCTThermostatSimulator (net::ComputationGraph* policy) {
		init (&t_, policy);
	}
};

#endif /* THERMOSTAT_H_ */
