#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <iostream>
#include <iterator>
using namespace std;

#include "pole.h"
#include "logger.h"
#include "system.h"
#include "agent.h"
#include "monte-carlo.h"
#include "sarsa.h"
#include "qlearning.h"
#include "sarsa-lambda.h"

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

Agent *CreatorAgent(AlgorithmType algorithm_t, PolicyType policy_type, bool train)
{
	switch (algorithm_t) {
	case AT_MonteCarlo: return new MonteCarloAgent(policy_type, !train);
	case AT_Sarsa: return new SarsaAgent(policy_type, !train);
	case AT_QLearning: return new QLearningAgent(policy_type, !train);
	case AT_SarsaLambda: return new SarsaLambdaAgent(policy_type, !train);
	default: return 0;
	}
}

void set_random_seed(int seed)
{
	srand(seed);
	srand48(seed);
}

int main(int argc, char **argv) {
	AlgorithmType algorithm_type = AT_None;
	PolicyType policy_type = PT_None;
	bool train = false;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
	            				("help,h", "produce help message")
	            				("train,t", "set as train mode")
	            				("algorithm,a", po::value<char>(), "set algorithm type m|s|q|l")
	            				("policy,p", po::value<char>(), "set policy type r|g|e|s")
	            				;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << desc << "\n";
			return 0;
		}

		if (vm.count("train")) {
			train = true;
		}

		if (!vm.count("algorithm") || !vm.count("policy")) {
			cout << desc << "\n";
			return 1;
		}

		switch (vm["algorithm"].as<char>()) {
		case 'm': algorithm_type = AT_MonteCarlo; break;
		case 's': algorithm_type = AT_Sarsa; break;
		case 'q': algorithm_type = AT_QLearning; break;
		case 'l': algorithm_type = AT_SarsaLambda; break;
		default: cout << desc << "\n"; return 1;
		}

		switch (vm["policy"].as<char>()) {
		case 'r': policy_type = PT_Random; break;
		case 'g': policy_type = PT_Greedy; break;
		case 'e': policy_type = PT_EpsilonGreedy; break;
		case 's': policy_type = PT_Softmax; break;
		default: cout << desc << "\n"; return 1;
		}
	}
	catch(exception& e) {
		cerr << "error: " << e.what() << "\n";
		return 1;
	}
	catch(...) {
		cerr << "Exception of unknown type!\n";
		return 1;
	}

	set_random_seed(getpid());

	Agent *agent = CreatorAgent(algorithm_type, policy_type, train);

	if (!train) { //test
		Logger logger("cart-pole.rcg");
		double reward = System().simulate(*agent, true, & logger);
		cout << "Reward: " << reward << endl;
	}
	else { //train
		const int episodes = 1024;
		double rewards = 0.0;
		int loops = episodes;

		do {
			rewards += System().simulate(*agent, false);
		} while(loops--);

		cout << rewards / double(episodes) << endl;
	}

	delete agent;

	return 0;
}
