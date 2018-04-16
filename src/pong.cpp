#include "qlearner/qlearner.hpp"
#include "qlearner/stats.h"
#include "qlearner/action.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <boost/lexical_cast.hpp>

#include "config.h"
#include "net/fc_layer.hpp"
#include "net/relu_layer.hpp"
#include "net/tanh_layer.hpp"
#include "net/solver.hpp"
#include "net/rmsprop.hpp"
#include "net/network.hpp"

#include <random>

using namespace net;
using namespace qlearn;

const float BAT_SIZE = 0.05;

Vector dconv(float val, float min, float max, int steps, Vector result)
{
	float p = (val - min) / (max - min);
	int rp = std::min(steps-1, std::max(0, int(steps * p)));
	result = Matrix::Zero(steps, 1);
	result[rp] = 1;
	return std::move(result);
}

struct PongGame
{
	float ballx;
	float bally;
	float bvx;
	float bvy;
	float posy;
	
	PongGame()
	{
		
	}

	void reset()
	{
		ballx = 0.6;
		bally = (rand() % 101) / 100.f;
		bvx = 1;
		bvy = (rand() % 101 - 50) / 20.f;
		posy = (rand() % 101) / 100.f;
	}

	float step( int ac )
	{
		ballx += 0.01*bvx;
		bally += 0.01*bvy;

		if( ac == 1 )
			posy += 0.025;
		else if(ac == 2)
			posy -= 0.025;

		if(bally > 1)
		{
			bally = 2 - bally;
			bvy *= -1;
		}

		if(bally < 0)
		{
			bally = -bally;
			bvy *= -1;
		}

		if( ballx > 1.0 )
		{
			if( std::abs(bally - posy) < BAT_SIZE )
				return 1;
			else
				return -1;
		}

		return 0;
	}

	const Vector& data() const
	{
		vec.resize (5);
		vec[0] = ballx;
		vec[1] = bally;
		vec[2] = posy;
		vec[3] = bvx;
		vec[4] = bvy;
		/*vec.resize(30);
		auto d1 = dconv(bally, 0, 1, 15, std::move(dccache1));
		auto d2 = dconv(posy, 0, 1, 15, std::move(dccache2));
		vec << d1, d2;
		dccache1 = std::move(d1);
		dccache2 = std::move(d2);*/
		return vec;
	}
	
private:
	mutable Vector vec;
	mutable Vector dccache1;
	mutable Vector dccache2;
};

void build_image(const qlearn::QLearner& l);

std::mutex mTargetNet;
std::atomic<bool> evaluate(false);
std::atomic<bool> run(true);

void learn_thread( Network& target_net, ComputationGraph& graph )
{
	Config config(5, 3, 2000000);
	config.epsilon_steps(2000000).update_interval(10000).batch_size(32).init_memory_size(10000).init_epsilon_time(100000)
		.discount_factor(0.98);
	
	Network network;
	network << FcLayer(Matrix::Random(10, 5).array() / 5);
	network << TanhLayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(10, 10).array() / 5);
	network << TanhLayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(3, 10).array() / 5);
	network << TanhLayer(Matrix::Zero(3, 1));
	
	qlearn::QLearner learner( config, std::move(network) );
	
	auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.0005, 0.001));
	RMSProp* rmsprop = prop.get();
	Solver solver( std::move(prop) );
	
	std::fstream rewf("reward.txt", std::fstream::out);

	PongGame game;
	game.reset();
	int ac = 0;
	int games = 0;
	auto last_time = std::chrono::high_resolution_clock::now();
//	bool run = true;
	
	learner.setCallback( [&](const QLearner& learner, const Stats& stats ) 
	{
		std::cout << games << ": " << learner.getCurrentEpsilon() << "\n";
		std::cout << stats.getSmoothReward() << " (" <<  stats.getSmoothQVal() << ", " << stats.getSmoothMSE() << ")\n";
		rewf << stats.getSmoothReward() << "\t" <<  stats.getSmoothQVal() << "\t" << stats.getSmoothMSE() << " " << learner.getCurrentEpsilon()  << "\n";
		rewf.flush();
//		std::cout << learner.getNumberLearningSteps() << "\n";
		build_image(learner);
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::high_resolution_clock::now() - last_time).count() << " ms\n";
		last_time = std::chrono::high_resolution_clock::now();
//		std::cout << Eigen::internal::malloc_counter() << "\n";
		std::cout << " - - - - - - - - - - \n";
		std::lock_guard<std::mutex> lck(mTargetNet);
		target_net = learner.network().clone();
		graph = ComputationGraph(target_net);
		evaluate = true;
	} );

	while(run)
	{
		/*if(games < 1000)
		{
			rmsprop->setRate(0.001);
		} else if ( games < 5000 )
		{
			rmsprop->setRate(0.005);
		} else
		{
			rmsprop->setRate(0.0001);
		}*/

		float r = game.step(ac);
		ac = learner.learn_step( game.data(), r, game.ballx >= 1, solver );
		if( game.ballx >= 1.0 )
		{
			game.reset();
			games++;
		}
	}
}

extern int status;
/*int main()
{
	Network network;
	ComputationGraph graph(network);
	std::thread learner( learn_thread, std::ref(network), std::ref(graph));
	learner.detach();
	
	PongGame game;
	game.reset();
	int games = 0;

	std::fstream evl("test.txt", std::fstream::out);
	
	int step = 0;
	while(true)
	{
		//step++;
		//float v;
		//{
		//	std::lock_guard<std::mutex> lck(mTargetNet);
		//	auto ac = getAction(graph, game.data());
		//	game.step(ac.id);
		//	v = ac.score;
		//}

		//if( game.ballx > 1.0 )
		//{
		//	game.reset();
		//	games++;
		//	sleep(100);
		//}
		
		if(evaluate)
		{
			Network copy = network.clone();
			ComputationGraph graph(copy);
			float reward = 0;
			for(int g = 0; g < 200; ++g)
			{
				PongGame game;
				game.reset();
				for(int s = 0; s < 100; ++s)
				{
					auto ac = getAction(graph, game.data());
					reward += game.step(ac.id);
					if( game.ballx > 1.0 ) break;
				}
			}
			std::cout << reward << "\n";
			evl << reward << "\n";
			evl.flush();
			evaluate = false;
			sleep (10);
		}
	}
	run = false;
} */

void build_image(const QLearner& l)
{

}


