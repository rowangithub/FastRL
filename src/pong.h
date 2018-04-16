#ifndef PONG_H_
#define PONG_H_

#include <cstdlib>
#include <cmath>
#include <iostream>

#include "utils.h"
#include "Game.h"
#include "boost/tuple/tuple.hpp"

const float BAT_SIZE = 0.05;

class Logger;

class Pong : public Game {
private:
	double ballx;
	double bally;
	double bvx;
	double bvy;
	double posy;
	
public:
	Pong() {
		reset();
	}

	bool terminate() const {
		if( ballx > 1.0 && std::abs(bally - posy) < BAT_SIZE ) {
				return true;
		}
		return false;
	}

	bool fail() const {
		if( ballx > 1.0 && std::abs(bally - posy) >= BAT_SIZE) {
			return true;
		}
		return false;
	}

	double measureFailure() const {
		return -1.0 * std::abs(bally - posy);
	}

	void perturbation() { 
		ballx = 0.6;
		bally = (rand() % 101) / 100.f;
		bvx = 1;
		bvy = (rand() % 101 - 50) / 20.f;
		posy = (rand() % 101) / 100.f;
    }

	void reset()
	{
		ballx = 0.6;
		bally = 0;
		bvx = 1;
		bvy = 0;
		posy = 0;
	}

	void step( int ac )
	{
		ballx += 0.01*bvx;
		bally += 0.01*bvy;

		if( ac == 1 )
			posy += 0.025;
		else if(ac == 2)
			posy -= 0.025;

		if(bally > 1)
		{
			bally = 2 - bally;
			bvy *= -1;
		}

		if(bally < 0)
		{
			bally = -bally;
			bvy *= -1;
		}
	}

    std::vector<double> getGameState () {
    	std::vector<double> data;
    	data.push_back(ballx);
    	data.push_back(bally);
    	data.push_back(posy);
    	data.push_back(bvx);
    	data.push_back(bvy);
		return data;          
	}

    void setGameState (std::vector<double> st) {
		ballx = st[0];
		bally = st[1];
		posy = st[2];
		bvx = st[3];
		bvy = st[4]; 
    }

    void printGameState() {
		std::cout << "(" << ballx << ", " << bally << ", " << posy << ", " << bvx << ", " << bvy << ")\n";
	}

    int actions() {return 3;}
    int inputs() {return 5;}
};

class UCTPongSimulator : public UCTGameSimulator {
public:
	// Game engine
    Pong pong_;

    UCTPongSimulator () {
		init (&pong_);
	}

	UCTPongSimulator (net::ComputationGraph* policy) {
		init (&pong_, policy);
	}
};

#endif /* PONG_H_ */
