#include "pole.h"
#include "logger.h"
#include "system.h"
#include "agent.h"
#include "monte-carlo.h"
#include "sarsa.h"
#include "qlearning.h"
#include "sarsa-lambda.h"
#include "neural-agent.h"

/**
 * TODO:
 * 		1������״̬���� - done���������ȿ��Եõ���ϸ�µĲ���
 * 		2���Ľ��ر����� - done���ر�����Խ�ܾ�ȷ���֣�״̬��������Խ������ѧϰ
 *      3�����ӷ������� - ������
 *      4����֤���ϸ������gamma��alpha�ȶ�ѧϰ���̵�Ӱ�죺
 *      	a����gamma=1��Ч���ǳ���
 *      	b��q���ʼ��Ϊ0Ч���������ʼ����
 *      5�����rcg��ʽ���� - done
 *      6��ȡ��|theta|���� - done����ѧϰ������
 *      7���������� - done��Ĭ��������
 *      8��״̬�Ƿ���Ҫ����x? - ȷ���޹�
 */

using namespace std;

Agent *CreatorAgent(AgentType agent_t, bool train)
{
	switch (agent_t) {
	case AT_MonteCarlo: return new MonteCarloAgent(!train);
	case AT_Sarsa: return new SarsaAgent(!train);
	case AT_QLearning: return new QLearningAgent(!train);
	case AT_SarsaLambda: return new SarsaLambdaAgent(!train);
	case AT_Neuron: {
		NeuralAgent * agent = new NeuralAgent (!train);
		return agent;}
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
       const int episodes = 2048;

	double rewards = 0.0;

	bool tmp = agent->test();
	agent->set_test(true);

	for (int i = 0; i < episodes; ++i) {
		double r = System().simulate(*agent, false);
		rewards += r;
		
		cout << endl << "utility step = " << i << " r: " << r << endl;
	}

	agent->set_test(tmp);

	return rewards / double(episodes);
}

/*int main(int argc, char **argv) {
	bool train = false;
	AgentType agent_t = AT_None;

	int  opt;
	while ((opt = getopt(argc, argv, "tmnsql")) != -1) {
		switch (opt) {
		case 't': train = true; break;
		case 'm': agent_t = AT_MonteCarlo; break;
		case 's': agent_t = AT_Sarsa; break;
		case 'q': agent_t = AT_QLearning; break;
		case 'l': agent_t = AT_SarsaLambda; break;
		case 'n': agent_t = AT_Neuron; break;
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
		const int episodes = 100000;
		double rewards = 0.0;
		int loops = episodes;

		do {
			rewards = System().simulate(*agent, false);
			if (loops % 100 == 0)
				cout << endl << "train step = " << loops << " r: " << rewards << endl;
		} while(loops--);

		//evaluate policy
		cout << "Training completes\n" << endl;
		cout << utility(agent) << endl;
	}

	delete agent; //save learned table if necessarily

	return 0;
}*/
