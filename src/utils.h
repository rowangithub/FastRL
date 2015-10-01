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
#include <ctime>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

static const double FLOAT_EPS = 1.0e-6;
static const double one_degree = 2 * M_PI / 360.0;

inline double irand(const double & min, const double & max)
{
	return min + (max - min) * drand48();
}

inline double normal_dist() //mean: 0.0, sd: 1.0
{
    static boost::mt19937 rng(static_cast<unsigned>(getpid()));
    static boost::normal_distribution<> nd(0.0, 1.0);
    static boost::variate_generator<boost::mt19937, boost::normal_distribution<> > var_nor(rng, nd);

    return var_nor();
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
