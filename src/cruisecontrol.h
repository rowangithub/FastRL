#ifndef DRIVE_H_
#define DRIVE_H_

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "utils.h"
#include "Game.h"
#include "boost/tuple/tuple.hpp"

class Logger;

class CruiseControl : public Game {
	static constexpr double time_step = 0.3;
	static constexpr double r = 2;

private:
	double x;
	double gamma;
	
public:
	double w = M_PI / 4;
	
	Drive() {
		reset();
	}

	bool terminate() const {
		if (inv_mode) {
			// Successfully terminates within the invariant.
			return (game_step >= 10 && (x > -0.9 && x < 0.9 && gamma > -0.79 && gamma < 0.79));
		} else {
			return false;
		}
	}

	bool fail() const {
		if (inv_mode) {
			// Unsuccessfully terminates without respecting the invariant.
			return game_step >= 10 && (x <= -0.9 || x >= 0.9 || gamma <= -0.79 || gamma >= 0.79);
		} else {
			return (x <= -2 || x >= 2);
		}
	}

	double measureFailure() const {
		if (inv_mode) {
			return -1.0;
		} else
			return -1.0;
	}

	void perturbation() { 
		if (inv_mode) {
			x = irand (-0.9, 0.9);
			gamma = irand (-0.79, 0.79);
		} else {
			x = irand(-1, 1);
			gamma = irand(-1*w, w);
		}
    }

	void reset()
	{
		if (inv_mode) game_step = 0;

		x = 0;
		gamma = 0;
	}

	void step( int ac )
	{
		if (inv_mode) game_step += 1;

		double ddx = r*sin(gamma);
		x += time_step * ddx;
		double ddgamma = 0;
		if (ac == 0) { // Drive Left
			ddgamma = -1 * w;
		}
		else if (ac == 1) { // Drive ahead
			ddgamma = 0;
		}
		else if (ac == 2) { // Drive Right
			ddgamma = w;
		}
		gamma += time_step * ddgamma;
	}

    std::vector<double> getGameState () {
    	std::vector<double> data;
    	data.push_back(x);
    	data.push_back(gamma);
		return data;          
	}

    void setGameState (std::vector<double> st) {
		x = st[0];
		gamma = st[1];
    }

    void printGameState() {
			std::cout << "(" << x << "," << gamma << ")\n";
	}

    int actions() {return 3;}
    int inputs() {return 2;}
};

class UCTDriveSimulator : public UCTGameSimulator {
public:
	// Game engine
    Drive t_;

    UCTDriveSimulator () {
		init (&t_);
	}

	UCTDriveSimulator (net::ComputationGraph* policy) {
		init (&t_, policy);
	}
};

#endif /* DRIVE_H_ */
