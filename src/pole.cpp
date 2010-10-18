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
	static const double g = 9.8;
	static const double l = 1.0;
	static const double mc = 1.0;
	static const double mp = 0.1;

	double F = (!action)? 0: ((action > 0)? 10.0 : -10.0);

	double costheta = cos(theta_);
	double sintheta = sin(theta_);

	double ddtheta = (g * sintheta + costheta * ((-F - mp * l * dtheta_ * dtheta_ * sintheta) / (mc + mp))) / (l * (4.0 / 3.0 - (mp * costheta * costheta) / (mc + mp)));
	double ddx = (F + mp * l * (dtheta_ * dtheta_ * sintheta - ddtheta * costheta)) / (mc + mp);

	x_  += time_step * dx_;
	dx_ += time_step * ddx;

	theta_ += time_step * dtheta_;
	dtheta_ += time_step * ddtheta;
}

void Pole::log(Logger *logger, int action)
{
	static const double cart_width = 0.2;
	static const double cart_height = 0.075;
	static const double pole_len = 1.0;

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
