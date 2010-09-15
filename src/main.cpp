#include "pole.h"
#include "logger.h"
#include "system.h"
#include "agent.h"
#include "qlearning.h"

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
