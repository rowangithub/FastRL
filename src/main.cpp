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

using namespace std;

Agent *CreatorAgent(AgentType agent_t, bool train)
{
	switch (agent_t) {
	case AT_MonteCarlo: return new MonteCarloAgent(!train);
	case AT_Sarsa: return new SarsaAgent(!train);
	case AT_QLearning: return new QLearningAgent(!train);
	case AT_SarsaLambda: return new SarsaLambdaAgent(!train);
	default: return 0;
	}
}

void set_random_seed(int seed)
{
	srand(seed);
	srand48(seed);
}

void usage(const char *progname) {
	cerr << "Usage:\n\t" << progname << " [-t|m|s|q|l]\n"
			<< "Options:\n"
			<< "\t-t\ttrain mode\n"
			<< "\t-m\tuse monte-carlo method\n"
			<< "\t-s\tuse sarsa method\n"
			<< "\t-q\tuse qlearning method\n"
			<< "\t-l\tuse sarsa(lambda) method"
			<< std::endl;
}

double utility(Agent *agent)
{
    const int episodes = 5120;

	double rewards = 0.0;

	bool tmp = agent->test();
	agent->set_test(true);

	for (int i = 0; i < episodes; ++i) {
		rewards += System().simulate(*agent, false);
	}

	agent->set_test(tmp);

	return rewards / double(episodes);
}

int main(int argc, char **argv) {
	bool train = false;
	AgentType agent_t = AT_None;

	int  opt;
	while ((opt = getopt(argc, argv, "tmsql")) != -1) {
		switch (opt) {
		case 't': train = true; break;
		case 'm': agent_t = AT_MonteCarlo; break;
		case 's': agent_t = AT_Sarsa; break;
		case 'q': agent_t = AT_QLearning; break;
		case 'l': agent_t = AT_SarsaLambda; break;
		default: usage(argv[0]); exit(1);
		}
	}

	set_random_seed(getpid());

	Agent *agent = CreatorAgent(agent_t, train);

	if (!agent) {
		cerr << "Error: No learning method provided" << endl;
		usage(argv[0]);
		return 1;
	}

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

		//evaluate policy
		cout << utility(agent) << endl;
	}

	delete agent; //save learned table if necessarily

	return 0;
}
