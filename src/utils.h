/*
 * utils.h
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <cmath>
#include <cstdlib>

static const double FLOAT_EPS = 1.0e-6;
static const double one_degree = 2 * M_PI / 360.0;

inline double irand(const double & min, const double & max)
{
	return min + (max - min) * drand48();
}

inline double prob()
{
	return drand48();
}

inline double normalize_angle(double angle)
{
	while (angle <= -M_PI) {
		angle += 2.0 * M_PI;
	}

	while (angle > M_PI) {
		angle -= 2.0 * M_PI;
	}

	return angle;
}

#endif /* UTILS_H_ */
