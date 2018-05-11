#ifndef DRIVE_H_
#define DRIVE_H_

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "utils.h"
#include "Game.h"
#include "boost/tuple/tuple.hpp"

class Logger;

class Drive : public Game {
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
		return false;
	}

	bool fail() const {
		if( x <= -2 || x >= 2) {
			return true;
		}
		return false;
	}

	double measureFailure() const {
		return -1.0;
	}

	void perturbation() { 
		x = irand(-1, 1);
		gamma = irand(-1*w, w);
    }

	void reset()
	{
		x = 0;
		gamma = 0;
	}

	void step( int ac )
	{
		double ddx = -1*r*sin(gamma);
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
