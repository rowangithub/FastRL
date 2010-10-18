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
#include <algorithm>

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

template<typename _Tp>
inline const _Tp&
minmax(const _Tp& min_, const _Tp& x, const _Tp& max_)
{
    return std::min(std::max(min_, x), max_);
}

#endif /* UTILS_H_ */
