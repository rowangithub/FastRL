/*
 * pole.cpp
 *
 *  Created on: Sep 15, 2010
 *      Author: baj
 */

#include "pole.h"
#include "logger.h"

// Nonlinear control
// void Pole::step(int action)
// {
//     if (inv_mode) game_step += 1;

//     static const double f = 1.0;
// 	static const double g = 9.8;
// 	static const double l = 1.0;
// 	static const double mc = 1.0;
// 	static const double mp = 0.1;

// 	//double force = ((!action)? 0: ((action > 0)? f : -f));
// 	double force = ((action == 1)? 0: ((action > 1)? f : -f));

//     //if (action) {
//     //    force += irand(-f, f) * 0.5;
//     //}

// 	double costheta = cos(theta_);
// 	double sintheta = sin(theta_);

// 	double ddtheta = (g * sintheta + costheta * ((-force - mp * l * dtheta_ * dtheta_ * sintheta) / (mc + mp))) / (l * (4.0 / 3.0 - (mp * costheta * costheta) / (mc + mp)));
// 	double ddx = (force + mp * l * (dtheta_ * dtheta_ * sintheta - ddtheta * costheta)) / (mc + mp);

// 	x_  += time_step * dx_;
// 	dx_ += time_step * ddx;

// 	theta_ += time_step * dtheta_;
// 	dtheta_ += time_step * ddtheta;
// }

#define N 4

void multiplyMatrix(int m1, int m2, double* mat1, int n2, double* mat2, double* res)
{
    int x, i, j;
    for (i = 0; i < m1; i++) {
        for (j = 0; j < n2; j++) {
            *(res + i*n2 + j) = 0;
            for (x = 0; x < m2; x++) {
                *(res + i*n2 + j) += *(mat1 + i*m2 + x) *
                                     *(mat2 + x*n2 + j);
            }
        }
    }
}

void addMatrix(int m1, int m2, double* mat1, double* mat2, double* res)
{
    int i, j;
    for (i = 0; i < m1; i++)
    {
        for (j = 0; j < m2; j++)
        {
            *(res + i*m2 + j) = 0;
            *(res + i*m2 + j) += *(mat1 + i*m2 + j) + *(mat2 + i*m2 + j);
        }
    }
}

// Linear Cotrol
void Pole::step(int action)
{
	if (inv_mode) game_step += 1;
	// https://danielpiedrahita.wordpress.com/portfolio/cart-pole-control/
	// A = [0 1 0      0; ...
 	//     0 0 0.7164  0;...
 	//     0 0 0      1; ...
 	//     0 0 15.76 0];
	// B = [0;0.9755;0;1.46];
	// For a given plant model \dot(x(t)) = Ax(t) + Bu(t)
	// we want to find a control input u that controls the plant well

	double u = ((action == 1)? 0: ((action > 1)? 0.2 : -0.2));

	double A[N][N] = { {0, 1, 0, 0},
                    {0, 0, 0.7164, 0},
                    {0, 0, 0, 1},
                    {0, 0, 15.76, 0}};
 
    double B[N][1] = { {0},
                    {0.9755},
                    {0},
                    {1.46}};

	
	double X[N][1] = {{x_}, {dx_}, {theta_}, {dtheta_}};
	double U[1][1] = {{u}};

	double res1[N][1];
	multiplyMatrix(N, N, A[0], 1, X[0], res1[0]);

	double res2[N][1];
	multiplyMatrix(N, 1, B[0], 1, U[0], res2[0]);

	double res3[N][1];
	addMatrix(N, 1, res1[0], res2[0], res3[0]);

	dx_ = res3[0][0];
	double ddx = res3[1][0];
	theta_ = res3[2][0];
	double ddtheta = res3[3][0];

	x_  += time_step * dx_;
	dx_ += time_step * ddx;
	theta_ += time_step * dtheta_;
	dtheta_ += time_step * ddtheta;
}

// ignore the input action but just use LQR control
// void Pole::step (int action) {
// 	// K = [-3.1623, -4.2691, 38.9192, 9.9633]
// 	double K[1][N] = {{-3.1623, -4.2691, 38.9192, 9.9633}};

// 	double A[N][N] = { {0, 1, 0, 0},
//                     {0, 0, 0.7164, 0},
//                     {0, 0, 0, 1},
//                     {0, 0, 15.76, 0}};
 
//     double B[N][1] = { {0},
//                     {0.9755},
//                     {0},
//                     {1.46}};

	
// 	double X[N][1] = {{x_}, {dx_}, {theta_}, {dtheta_}};

// 	double U[1][1];
// 	multiplyMatrix(1, N, K[0], 1, X[0], U[0]);

// 	double res1[N][1];
// 	multiplyMatrix(N, N, A[0], 1, X[0], res1[0]);

// 	double res2[N][1];
// 	multiplyMatrix(N, 1, B[0], 1, U[0], res2[0]);

// 	double res3[N][1];
// 	addMatrix(N, 1, res1[0], res2[0], res3[0]);

// 	dx_ = res3[0][0];
// 	double ddx = res3[1][0];
// 	theta_ = res3[2][0];
// 	double ddtheta = res3[3][0];

// 	x_  += time_step * dx_;
// 	dx_ += time_step * ddx;
// 	theta_ += time_step * dtheta_;
// 	dtheta_ += time_step * ddtheta;
// }

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
