#ifndef VERIFY_H_
#define VERIFY_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <random>

Vector vec;

typedef std::vector<double> gamestate;

// Encode game state using vector
const Vector& data(gamestate cpst)
{
	vec.resize (cpst.size());
	for (int i = 0; i < cpst.size(); i++) {
		vec[i] = cpst[i];
	}
	return vec;
}

void display(gamestate cpst)
{	
	for (int i = 0; i < cpst.size(); i++) {
		if (i != cpst.size() - 1)
			std::cout << cpst[i] << ",";
		else
			std::cout << cpst[i] << "\n";
	}
}

void display(gamestate cpst, int ac)
{	
	for (int i = 0; i < cpst.size(); i++) {
		if (i != cpst.size() - 1)
			std::cout << cpst[i] << ",";
		else
			std::cout << cpst[i] << " : ";
	}
	std::cout << ac << "\n";
}

void render(std::fstream& out, gamestate cpst, int ac)
{
	for (double d : cpst) {
		out << d << ",";
	}
	out << "Class" << ac << "\n";
}

#endif /* VERIFY_H_ */