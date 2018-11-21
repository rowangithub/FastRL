#ifndef CONFIG_H_
#define CONFIG_H_

#include <eigen3/Eigen/Dense>
#include <vector>

typedef float number_t;

using Matrix = Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<number_t, Eigen::Dynamic, 1>;

// fwd declarations
namespace net 
{
	class ILayer;
	class Solver;
	class ComputationNode;
	class Network;
}

// manage the number of python embedding instances
extern int python_mutex;

#endif