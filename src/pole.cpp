#include <iostream>
#include<fstream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <map>
#include <iomanip>
#include <algorithm>
#include <cassert>

#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"
#include "boost/tuple/tuple_io.hpp"

#include "logger.h"

/**
 * TODO:
 * 		1、增加状态粒度 - done，增加粒度可以得到更细致的策略
 * 		2、改进回报函数 - done，回报函数越能精确区分（状态、动作）越有利于学习
 *      3、增加泛化能力 - 不用了
 *      4、验证以上各项包括gamma和alpha等对学习过程的影响：
 *      	a、令gamma=1后，效果非常好
 *      	b、q表初始化为0效果较随机初始化好
 *      5、添加rcg格式动画 - done
 *      6、取消|theta|限制 - done，但学习很困难
 *      7、增加噪音 - done，默认有噪音
 *      8、状态是否不需要考虑x? - 确认无关
 */

using namespace std;

static const double FLOAT_EPS = 1.0e-6;
static const double one_degree = 2 * M_PI / 360.0;

inline double irand(const double & min, const double & max)
{
	return min + (max - min) * drand48();
}

class State {
public:
	State(double dx = 0.0, double theta = 0.0, double dtheta = 0.0) {
		this->dx() = dx * 100.0; //精确到厘米
		this->theta() = theta / one_degree; //精确到一度
		this->dtheta() = dtheta / one_degree; //精确到一度
	}

	int & dx() { return data_.get<0>(); }
	int & theta() {	return data_.get<1>(); }
	int & dtheta() { return data_.get<2>(); }

	const int & dx() const { return data_.get<0>(); }
	const int & theta() const {	return data_.get<1>(); }
	const int & dtheta() const { return data_.get<2>(); }

	bool operator<(const State & o) const {
		return data_ < o.data_;
	}

	bool operator==(const State & o) const {
		return data_ == o.data_;
	}

	friend ostream &operator<<(ostream & os, const State & o) {
		return os << o.data_;
	}

	friend istream &operator>>(istream & is, State & o) {
		return is >> o.data_;
	}

private:
	boost::tuples::tuple<int, int, int> data_;
};

class QTable {
	typedef boost::tuples::tuple<State, int> state_action_pair_t;

public:
	double & operator[](const boost::tuples::tuple<State, int> & key) {
		return table_[key]; //default initialized to zeros
	}

	void save(const char *file_name) const {
		ofstream fout(file_name);

		if (fout.good()) {
			for (map<state_action_pair_t, double>::const_iterator it = table_.begin(); it != table_.end(); ++it) {
				fout << it->first << " " << setprecision(13) << it->second << endl;
			}
		}

		fout.close();
	}

	void load(const char *file_name) {
		ifstream fin(file_name);

		if (fin.good()) {
			state_action_pair_t state_action_pair;
			double qvalue;

			while (!fin.eof()) {
				fin >> state_action_pair >> qvalue;
				table_[state_action_pair] = qvalue;
			}
		}

		fin.close();
	}

private:
	map<state_action_pair_t, double> table_;
};

class Pole {
public:
	Pole() {
		reset();
	}

	const double & x() const { return x_; }
	const double & dx() const { return dx_; }
	const double & theta() const {	return theta_; }
	const double & dtheta() const { return dtheta_; }

	bool fail() const {
		return fabs(theta_) > 10.0 * one_degree || fabs(dx_) > 1.0;
	}

	void perturbation() { //微小扰动 - 模拟人放置杆子
		dtheta_ = irand(-one_degree, one_degree);
	}

	void reset() {
		x_ = 0.0;
		dx_ = 0.0;
		theta_ = 0.0;
		dtheta_ = 0.0;
	}

	void step(int action) {
		/*** Parameters for simulation ***/
		static const double GRAVITY = 9.8;
		static const double MASSCART = 1.0;
		static const double MASSPOLE = 0.1;
		static const double TOTAL_MASS = (MASSPOLE + MASSCART);
		static const double LENGTH = 0.5;        /* actually half the pole's length */
		static const double POLEMASS_LENGTH = (MASSPOLE * LENGTH);
		static const double FORCE_MAG = 10.0;
		static const double TAU = 0.02;          /* seconds between state updates */
		static const double FOURTHIRDS = 1.3333333333333;

		double force = (!action)? 0: ((action > 0)? FORCE_MAG : -FORCE_MAG);

		force += irand(-FORCE_MAG, FORCE_MAG) * 0.1; //动作误差

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
	}

	/**
	 * state signal
	 */
	State get_signal() {
		return State(dx_, theta_, dtheta_);
	}

	void print_state(int step) {
		cout << "Step " << step << ": " <<
				x_ << " " << dx_ << " " << theta_ << " " << dtheta_;
	}

	void log(Logger *logger, int action) {
		static const double cart_width = 0.2;
		static const double cart_height = 0.075;
		static const double pole_len = 0.5;

		//cart-pole坐标系
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

private:
	double x_;
	double dx_;
	double theta_;
	double dtheta_;
};

class Agent {
public:
	virtual int plan(const State &) {
		return 0;
	}

	virtual double learn(const State &, int, double, const State &) {
		return 0.0;
	}

	virtual void fail(const State &, int) {

	}
};

class LeftAgent: public Agent {
public:
	int plan(const State &) {
		return -1;
	}
};

class RightAgent: public Agent {
public:
	int plan(const State &) {
		return 1;
	}
};

class RandomAgent: public Agent {
public:
	virtual int plan(const State &) {
		double prob = drand48();

		if (prob < 1 / 3.0) {
			return -1;
		}
		else if (prob > 2 / 3.0) {
			return 1;
		}
		else {
			return 0;
		}
	}
};

/**
 * Model-free QLearning Agent
 */
class QLearningAgent: public RandomAgent {
private:
	static const double alpha = 0.5;
	static const double gamma = 1.0 - 1.0e-6;

public:
	QLearningAgent(const double epsilon = 0.01, const bool test = false): epsilon_(epsilon), test_(test) {
		load();
	}

	~QLearningAgent() {
		if (!test_) {
			save();
		}
	}

	double & qvalue(const State & state, int action) {
		return qtable_[boost::tuples::make_tuple(state, action)];
	}

	int plan(const State & state) {
		if (drand48() < epsilon_) {
			return RandomAgent::plan(state);
		}
		else {
			return this->greedy(state);
		}
	}

	int greedy(const State & state) {
		vector<int> actions;
		double max = -10000.0;

		for (int i = -1; i <= 1; ++i) {
			double q = qvalue(state, i);
			if (q > max) {
				max = q;
				actions.clear();
				actions.push_back(i);
			}
			else if (q > max - FLOAT_EPS) {
				actions.push_back(i);
			}
		}

		if (!actions.empty()) {
			random_shuffle(actions.begin(), actions.end());
			return actions.front();
		}
		else {
			assert(0);
			return RandomAgent::plan(state);
		}
	}

	double learn(const State & pre_state, int pre_action, double reward, const State & state) {
		int action = greedy(state);
		double & u = qvalue(pre_state, pre_action);
		double v = qvalue(state, action);

		double error = alpha * (reward + gamma * v - u);

		u = u + error;

		return error;
	}

	void fail(const State & state, int action) {
		qvalue(state, action) = -10.0;
	}

	void save() {
		qtable_.save("qtable.txt");
	}

	void load() {
		qtable_.load("qtable.txt");
	}

private:
	QTable qtable_;
	const double epsilon_;
	const bool test_;
};

class System {
public:
	System() {
		pole_.perturbation();
	}

	void reset() {
		pole_.reset();
	}

	double get_reward(int action) { //评价范围 [-2.0, 2.0]
		return cos(pole_.theta()) + cos(pole_.dtheta()) - fabs(pole_.dx()) - abs(action); //以保持不动为最佳
	}

	double simulate(Agent & agent, bool verbose = true, Logger *logger = 0) {
		int step = 0;
		double rewards = 0.0;
		double mse = 0.0;
		State state = pole_.get_signal();

		do {
			step += 1;
			int action = agent.plan(state);

			if (verbose) {
				pole_.print_state(step);
				cout << " | State: " << pole_.get_signal() << " | Action: " << action;
			}

			if (logger) {
				pole_.log(logger, action);
				logger->Flush();
			}

			pole_.step(action);

			if (pole_.fail()) {
				 agent.fail(state, action); //failure state - 区别失败状态跟一般未知状态（未知状态初始化为零）
				 if (verbose) {
					 cout << " | Failure" << endl;
				 }
				 step += 1;
				 break;
			}

			State pre_state = state;
			state = pole_.get_signal();

			double reward = get_reward(action);
			double error = agent.learn(pre_state, action, reward, state);
			mse += error * error;

			if (verbose) {
				cout << " | Reward: " << reward << endl;
			}

			rewards += reward;
		} while(1);

		if (verbose) {
			pole_.print_state(step);
			cout << " | State: " << pole_.get_signal() <<  " | The End" << endl;
			cout << "MSE: " << mse << endl;
		}

		if (logger) {
			pole_.log(logger, 0);
			logger->Flush();
		}

		return rewards;
	}

private:
	Pole pole_;
};

void usage(const char *progname) {
	cout << "Usage:\n\t" << progname << " [-t] [-s seed] [-d]\n"
			<< "Options:\n"
			<< "\t-t\ttrain mode\n"
			<< "\t-s\tset random seed\n"
			;
}

int main(int argc, char **argv) {
	int seed = getpid();
	bool train = false;

	int  opt;
	while ((opt = getopt(argc, argv, "dts:")) != -1) {
		switch (opt) {
		case 't': train = true; break;
		case 's': seed = atoi(optarg); break;
		default: usage(argv[0]); exit(1);
		}
	}

	srand48(seed);

	if (!train) { //test
		QLearningAgent agent(-0.0, true);
		Logger logger("cart-pole.rcg");
		double reward = System().simulate(agent, true, & logger);
		cout << "Reward: " << reward << endl;
	}
	else { //train
		const int episodes = 1024;

		QLearningAgent agent;

		double rewards = 0.0;
		int loops = episodes;

		do {
			rewards += System().simulate(agent, false);
		} while(loops--);

		cout << "#Avg Reward:\n" << rewards / double(episodes) << endl;
	}

	return 0;
}
