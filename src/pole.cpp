/*
 * pole.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "pole.h"
#include "logger.h"

void Pole::step(int action)
{
	/*** Parameters for simulation ***/
	static const double GRAVITY = 9.8;
	static const double MASSCART = 1.0;
	static const double MASSPOLE = 0.1;
	static const double TOTAL_MASS = MASSPOLE + MASSCART;
	static const double LENGTH = 0.5;        /* actually half the pole's length */
	static const double POLEMASS_LENGTH = (MASSPOLE * LENGTH);
	static const double FORCE_MAG = 10.0;
	static const double TAU = 0.02;          /* seconds between state updates */
	static const double FOURTHIRDS = 1.3333333333333;

	double force = (!action)? 0: ((action > 0)? FORCE_MAG : -FORCE_MAG);

	double costheta = cos(theta_);
	double sintheta = sin(theta_);

	double temp = (force + POLEMASS_LENGTH * dtheta_ * dtheta_ * sintheta) / TOTAL_MASS;
	double thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS));
	double xacc  = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

	/*** Update the four state variables, using Euler's method. ***/
	x_  += TAU * dx_;
	dx_ += TAU * xacc;

	theta_ += TAU * dtheta_;
	dtheta_ += TAU * thetaacc;

	theta_ = normalize_angle(theta_);
}

void Pole::log(Logger *logger, int action)
{
	static const double cart_width = 0.2;
	static const double cart_height = 0.075;
	static const double pole_len = 0.5;

	//cart-pole×ø±êÏµ
	Logger::Rectangular cart(x_ - cart_width * 0.5, x_ + cart_width * 0.5, cart_height,	0.0);
	Logger::Vector pole_top = Logger::Vector(x_ + pole_len * sin(theta_), cart_height + pole_len * cos(theta_));
	Logger::Vector pole_bottom = Logger::Vector(x_ , cart_height);

	static const double x_scale = 250.0;
	static const double y_scale = -250.0;

	logger->Scale(x_scale, y_scale);
	logger->Focus(Logger::Vector(x_, 0.0));
	logger->LogRectangular(cart, Logger::Purple);
	logger->LogLine(pole_bottom, pole_top, Logger::Yellow, 0);

	if (action == 1) {
		Logger::Vector indicator = (cart.TopRightCorner() + cart.BottomRightCorner()) * 0.5;
		logger->AddPoint(indicator, 0, Logger::White);
	}
	else if (action == -1) {
		Logger::Vector indicator = (cart.TopLeftCorner() + cart.BottomLeftCorner()) * 0.5;
		logger->AddPoint(indicator, 0, Logger::White);
	}
}
