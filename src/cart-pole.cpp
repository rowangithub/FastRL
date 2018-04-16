#include "qlearner/qlearner.hpp"
#include "qlearner/stats.h"
#include "qlearner/action.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <limits.h>

#include "config.h"
#include "net/fc_layer.hpp"
#include "net/relu_layer.hpp"
#include "net/tanh_layer.hpp"
#include "net/solver.hpp"
#include "net/rmsprop.hpp"
#include "net/network.hpp"

#include "dt.h"

#include "pole.h"

#include "uct-agent.h"

#include <random>

using namespace net;
using namespace qlearn;

const float BAT_SIZE = 0.05;

typedef boost::tuples::tuple<double, double, double, double> polestate;

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

Vector vec;

// Encode game state using vector
const Vector& data(boost::tuples::tuple<double, double, double, double> cpst)
{
	vec.resize (4);
	vec[0] = cpst.get<0>();
	vec[1] = cpst.get<1>();
	vec[2] = cpst.get<2>();
	vec[3] = cpst.get<3>();
	return vec;
}

void display(boost::tuples::tuple<double, double, double, double> cpst)
{	
	std::cout << cpst.get<0>() << "," << cpst.get<1>() << "," 
		<< cpst.get<2>() << "," << cpst.get<3>() << "\n";
}

void display(boost::tuples::tuple<double, double, double, double> cpst, int ac)
{	
	std::cout << cpst.get<0>() << "," << cpst.get<1>() << "," 
		<< cpst.get<2>() << "," << cpst.get<3>() << " : " << ac << "\n";
}

void render(std::fstream& out, boost::tuples::tuple<double, double, double, double> cpst, int ac)
{
	out << cpst.get<0>() << "," << cpst.get<1>() << "," 
		<< cpst.get<2>() << "," << cpst.get<3>() << ",";

	if (ac == 0)
		out << "Left\n";
	else if (ac == 2)
		out << "Right\n";
	else 
		out << "Stay\n";
}

void build_image(const qlearn::QLearner& l);

std::mutex mTargetNet;
std::atomic<bool> evaluate(false);
std::atomic<bool> run(true);

void learn_thread( Network& target_net, ComputationGraph& graph )
{
	Config config(4, 3, 2000000);
	config.epsilon_steps(2000000).update_interval(10000).batch_size(32).init_memory_size(10000).init_epsilon_time(100000)
		.discount_factor(0.98);
	
	Network network;
	network << FcLayer(Matrix::Random(10, 4).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(10, 10).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(3, 10).array() / 5);
	//network << TanhLayer(Matrix::Zero(3, 1));
	
	qlearn::QLearner learner( config, std::move(network) );
	
	auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.0005, 0.001));
	RMSProp* rmsprop = prop.get();
	Solver solver( std::move(prop) );
	
	std::fstream rewf("reward.txt", std::fstream::out);

	Pole game;
	game.reset ();
	game.perturbation ();

	int ac = 1;
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

		game.step(ac-1);
		float r = 0;
		bool failed = game.fail();
		if (failed) r = -1;

		ac = learner.learn_step( data (game.getState()), r, failed, solver );
		if( failed )
		{
			game.reset();
			game.perturbation ();
			games++;
		}
	}
}

extern int status;
void agentPlay (std::string filename);
void networkToJson (std::string filename);
void abstraction_check (std::string filename);
void cegis_train();
int main(int argc, char** argv)
{
	// Network network;
	// network << FcLayer(Matrix::Random(10, 4).array() / 5);
	// network << ReLULayer(Matrix::Zero(10, 1));
	// network << FcLayer(Matrix::Random(10, 10).array() / 5);
	// network << ReLULayer(Matrix::Zero(10, 1));
	// network << FcLayer(Matrix::Random(3, 10).array() / 5);
	
	// ComputationGraph graph(network);

	// Vector vec;
	// vec.resize(4);
	// vec[0] = -0.469213;
	// vec[1] = -0.0971917;
	// vec[2] = -0.00305836;
	// vec[3] = 0.0654995;

	// auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.0005, 0.001));
	// RMSProp* rmsprop = prop.get();
	// Solver solver( std::move(prop) );
	// int error;
	// do {
	// 	error = 0;
	// 	const auto& result = graph.forward(vec);
	// 	int row, col;
	// 	result.maxCoeff(&row,&col);
	// 	error += (row == 2? 0 : 1);
	// 	std::cout << "Its original behavior on this is: " << row  << "\n";
	// 	for (int k = 0; k < result.size(); k ++)
	// 		std::cout << result[k] << " ";
	// 	std::cout << "\n";
	// 	Vector errorCache = Vector::Zero( result.size() );

	// 	//float delta = result[i->second] - target_value;
	// 	//errorCache[i->second] = delta;
	// 	for (int k = 0; k < result.size(); k++) {
	// 		if (k == 2) 
	// 			errorCache[k] = result[k] - 1.0;
	// 		else
	// 			errorCache[k] = result[k] - (-1.0);
	// 	}

	// 	graph.backpropagate(errorCache, solver);
	// 	network.update(solver);
	// } while (error > 0);

	if (argc > 1) {
		std::string param(argv[1]);

		if (param.compare ("uct") == 0) {
			UCTPoleSimulator* sim = new UCTPoleSimulator ();
		    UCTPoleSimulator* sim2 = new UCTPoleSimulator ();
		    UCT::UCTPlanner uct (sim2, -1, 1000, 1, 0.95);
		    int numGames = 1;

		    for (int i = 0; i < numGames; ++i) {
		        int steps = 0;
		        double r = 0;
		        sim->getState()->print(); cout << endl;
		        while (! sim->isTerminal()) {
		            steps++; //std::cout << "step : " << steps << "\n";
		            uct.setRootNode(sim->getState(), sim->getActions(), r, sim->isTerminal());
		            uct.plan();
		            UCT::SimAction* action = uct.getAction();

					uct.testTreeStructure();

		            r = sim->act(action);
		            sim->getState()->print(); cout << endl;
		        }
		        sim->reset();
		        cout << "Game:" << i << "  steps: " << steps << "  r: " << r << endl;
			}
		} 
		else if (param.compare ("cegis") == 0) {
			cegis_train();
		} 
		else {
			//Network agent;
			//agent.load (agentfile);
			//ComputationGraph graph(agent);

			//vec.resize (4);
			//vec[0] = -0.189315;
			//vec[1] = -0.00291353;
			//vec[2] = 0.00289438;
			//vec[3] = 0.00875231;


			//auto ac = getAction(graph, vec);
			//std::cout << "decision : " << ac.id-1 << "\n";

			//agentPlay (param);
			//networkToJson(agentfile);
			
			// -- doing abstraction refinement --
			abstraction_check (param);
		}
		return 0;
	}

	Network network;
	ComputationGraph graph(network);
	std::thread learner( learn_thread, std::ref(network), std::ref(graph));
	learner.detach();

	int games = 0;

	std::fstream evl("test.txt", std::fstream::out);
	
	int step = 0;
	while(true)
	{	
		if(evaluate)
		{
			Network copy = network.clone();
			ComputationGraph graph(copy);
			float reward = 0;
			int ts = 0;
			for(int g = 0; g < 200; ++g)
			{
				Pole game;
				game.reset();
				game.perturbation();
				for(int s = 0; s < 200; ++s)
				{
					auto ac = getAction(graph, data(game.getState()));
					game.step(ac.id-1);
					float currReward = 0;
					bool failed = game.fail();
					if (failed) currReward = -1;
					reward += currReward;
					ts ++;
					if( failed ) break;
				}
			}
			std::cout << (ts / 200.0) << "\n";
			if (ts / 200.0 == 200.0) {
				copy.save ("agent" + std::to_string(step) + ".network");
			}
			evl << reward << "\n";
			evl.flush();
			evaluate = false;
			step ++;
			sleep (10);
		}
	}
	run = false;
	return 0;
}

void build_image(const QLearner& l)
{

}

//===================  Test and Verification of learned neural model =================== 
// Collect samples for each class.
void agentSample (ComputationGraph& graph, int acts, std::vector<std::pair<polestate,int>>& keyset) {
	Pole game;
	game.reset();
	game.perturbation();
	int s = 0;
	int th = 0;
	int c = 0;
	while (1) {
		polestate st = game.getState();
		auto ac = getAction(graph, data(st));
		if (ac.id == th) {
			keyset.push_back(std::make_pair (st, ac.id));
			th++;
			c++;
		}
		game.step(ac.id-1);
		if (c == acts) break;
		bool failed = game.fail();
		s++;
		if( failed ) break;
	}
}

// test a learn model in file.
void agentPlay (std::string filename) {
	Network agent;
	agent.load (filename);
	ComputationGraph graph(agent);
	float reward = 0;
	int ts = 0;

	std::fstream evl("c5.data", std::fstream::out);
			
	for(int g = 0; g < 100; ++g)
	{
		Pole game;
		game.reset();
		game.perturbation();
		display(game.getState());
		int s = 0;
		while(true)
		{
			auto ac = getAction(graph, data(game.getState()));
			//std::cout << s << ": "; display (game.getState(), ac.id-1);
			game.step(ac.id-1);
			render (evl, game.getState(), ac.id);
			evl.flush ();
			float currReward = 0;
			bool failed = game.fail();
			if (failed) currReward = -1;
			reward += currReward;
			ts ++;
			s ++;
			if( failed ) {
				std::cout << "Maintained in " << s << " steps\n";
				break;
			}
		}
	}
	evl.close();
	std::cout << (ts / 100.0) << "\n";
}

bool agentPlay (Pole& game, ComputationGraph& graph, int bound) {
	int s = 0;
	while (s < bound) {
		auto ac = getAction(graph, data(game.getState()));
		game.step(ac.id-1);
		bool failed = game.fail();
		s++;
		if( failed ) break;
	}
	//std::cout << "agent played " << s << " step\n";
	return (s == bound);
}

// export a learned model
void networkToJson (std::string filename) {
	Network agent;
	agent.load (filename);
	agent.saveJson (filename + ".json");
}

// UCT advisor
int uct_advise (polestate st, bool terminal, ComputationGraph& graph) {
    UCTPoleSimulator* sim2 = new UCTPoleSimulator (&graph);
    UCT::UCTPlanner uct (sim2, -1, 1000, 1, 0.95);

    UCTPoleState* current = new UCTPoleState (st);
    uct.setRootNode (current, sim2->getActions(), 0, terminal);
    uct.plan();
    UCT::SimAction* action = uct.getAction();
    const UCTPoleAction* act = dynamic_cast<const UCTPoleAction*> (action);
    return (act->id)+1;
}

// Train NN with key points
bool key_gen (Pole& game, ComputationGraph& graph, int bound, std::vector<std::pair<polestate,int>>& keyset) {
	if (bound == 0)
		return true;

	// game should be in the very initial state.
	game.reset();
	game.perturbation();

	std::vector<std::pair<polestate,int>> path;
	int s = 0;
	while(s < bound)
	{
		//boost::tuples::tuple<double, double, double, double> 
		polestate st = game.getState();
		auto ac = getAction(graph, data(game.getState()));
		path.push_back (std::make_pair(st, ac.id));
		game.step(ac.id-1);
		display (st, ac.id-1);
		bool failed = game.fail();
		s ++;
		if( failed ) break;
	}
	std::cout << "how well is the trained network: " << s << " steps\n";
	if (s == bound) {
		// neural network is good enough so there is no need to improve it.
		return true;
	}
	else {
		for (std::vector<std::pair<polestate,int>>::reverse_iterator i = path.rbegin(); i != path.rend(); ++i) { 
			// Check how mc thinks about the best move.
			//std::cout << "from the " << s << " step in the counterexample\n";
			int ac = uct_advise(i->first, game.fail(), graph);
			if (ac == i->second) {
				// The mc and neural net agrees with each other.
			} else {
				std::vector<std::pair<polestate,int>> local_keyset;
				polestate st = i->first;
				game.setState (st);
				int i = 1;
				bool res = false;
				while (i <= bound - s) {
					local_keyset.push_back (std::make_pair (st, ac));
					game.step(ac-1);
					if (game.fail()) break;
					st = game.getState();
					//std::cout << "agent needs to play " << bound - s - i << " steps from "; display (st);
					res = agentPlay (game, graph, bound - s - i);
					game.setState(st);
					if (res) break;
					ac = uct_advise(st, game.fail(), graph);
					i++;
				}

				if (res) {
					std::cout << "we can fix the the counterexample path\n";
					keyset.insert(keyset.end(), local_keyset.begin(), local_keyset.end());
					return true;
				}
			}
			s--;
		}
		std::cout << "we cannot fix the counterexample path\n";
		return false;
	}
}

// CEGIS based training for controller synthesis.
void cegis_train() {
	Network network;
	network << FcLayer(Matrix::Random(10, 4).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(10, 10).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(3, 10).array() / 5);
	network << TanhLayer(Matrix::Zero(3, 1));
	
	ComputationGraph graph(network);

	Pole game;

	int bound = 200;

	std::vector<std::pair<polestate,int>> keyset;

	int g = 0;
	do {
		int size = keyset.size();
		bool fixable = key_gen (game, graph, bound, keyset);
		if (!fixable) {
			std::cout << "Should go back to reinforcement learning?\n";
			return;
		}
		if (keyset.size() == size) {
			g++;
		} else {
			g = 0;
			// Supervised learning to train the controller.
			auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.005, 0.001));
			RMSProp* rmsprop = prop.get();
			Solver solver( std::move(prop) );
			// First learn on the new keypoints
			int error;
			int currSize = keyset.size() - size;
			std::cout << "---------------------------------------------------------------\n";
			int iteri = 0;
			do {
				error = 0;
				int index = 0;	
				for (std::vector<std::pair<polestate,int>>::reverse_iterator i = keyset.rbegin(); 
						i != keyset.rend() && index < currSize; ++i) {
					//std::cout << "Enforcing the neural network to behave as: \n";
					//display (i->first, i->second-1);

					//float target_value = 1.0;
			
					const auto& result = graph.forward(data(i->first));
					int row, col;
					result.maxCoeff(&row,&col);
					error += (row == i->second? 0 : 1);
					//std::cout << "Its original behavior on this is: " << row - 1 << "\n";
					//for (int k = 0; k < result.size(); k ++)
					//	std::cout << result[k] << " ";
					//std::cout << "\n";
					Vector errorCache = Vector::Zero( result.size() );

					//float delta = result[i->second] - target_value;
					//errorCache[i->second] = delta;
					for (int k = 0; k < result.size(); k++) {
						if (k == i->second) 
							errorCache[k] = result[k] - 1.0;
						else
							errorCache[k] = result[k] - (-1.0);
					}

					graph.backpropagate(errorCache, solver);

					index ++;
				}
				network.update( solver );
				iteri ++;
				if (iteri % 10000 == 0)
					std::cout << "temporaral fix training error number: " << error << "\n";
			} while (error > 0 && iteri < 100000);
			std::cout << "fix training with error number: " << error << "\n";
			std::cout << "---------------------------------------------------------------\n";
			// Second learn on the whole keypoints
			iteri = 0;
			do {
				error = 0;
				for (std::vector<std::pair<polestate,int>>::iterator i = keyset.begin(); 
						i != keyset.end(); ++i) { 
					
					//float target_value = 1.0;
			
					const auto& result = graph.forward(data(i->first));
					int row, col;
					result.maxCoeff(&row,&col);
					error += (row == i->second? 0 : 1);

					Vector errorCache = Vector::Zero( result.size() );
					
					//float delta = result[i->second] - target_value;
					//errorCache[i->second] = delta;
					for (int k = 0; k < result.size(); k++) {
						if (k == i->second) 
							errorCache[k] = result[k] - 1.0;
						else
							errorCache[k] = result[k] - (-1.0);
					}

					graph.backpropagate(errorCache, solver);
				}
				network.update( solver );
				iteri++;
				//std::cout << "error number: " << error << "\n";
			} while (error > 0 && iteri < 10000);
			std::cout << "batch training error number: " << error << "\n";
		}
	} while (g < 5);
	network.save ("supervised_agent.network");
	std::cout << "Trained Successfully!\n";
}

// Verify an abstraction
bool ce_gen (Pole& game, ComputationGraph& graph, DT& dt, int bound, std::vector<std::pair<polestate,int>>& keyset, bool rec_flag) {
	if (bound == 0)
		return true;

	std::vector<std::pair<polestate,int>> path;
	int s = 0;
	while(s < bound)
	{
		//boost::tuples::tuple<double, double, double, double> 
		polestate st = game.getState();
		double*  dtst = new double[4];
		dtst[0] = st.get<0>();dtst[1] = st.get<1>();dtst[2] = st.get<2>();dtst[3] = st.get<3>();
		int ac = dt.predict (dtst, 4);
		delete [] dtst;
		path.push_back (std::make_pair(st, ac));
		game.step(ac-1);
			
		bool failed = game.fail();
		s ++;
		if( failed ) break;
	}
	if (!rec_flag)
		std::cout << "how well is the abstraction: " << s << " steps\n";
	if (s == bound) {
		// abstraction is fine so there is no counterexample
		return true;
	}
	else {
		for (std::vector<std::pair<polestate,int>>::reverse_iterator i = path.rbegin(); i != path.rend(); ++i) { 
			game.setState (i->first);
			// Check how neural net works.
			auto ac = getAction(graph, data(i->first));
			if (ac.id == i->second) {
				// The abstraction and neural net agrees with each other.
			} else {
				// Check if neural net's decision is good.
				bool res = agentPlay (game, graph, bound - s);
				game.setState(i->first);
				if (res) {
					keyset.push_back (std::make_pair (i->first, ac.id));
					game.step(ac.id-1);
					return ce_gen (game, graph, dt, bound - s - 1, keyset, true);
				}
			}
			s--;
		}
		return false;
	}
}

// abstract-refine checking of a neural model
void abstraction_check (std::string filename) {
	Network agent;
	agent.load (filename);
	std::string modelname = filename + ".json";
	agent.saveJson (modelname);
	ComputationGraph graph(agent);
	int g = 0;

	// Load abstraction from decision tree
	DT dt(modelname);
	std::vector<std::pair<polestate,int>> keyset;
	// Use keyset to refine the abstraction
	std::string datafile = filename+".data";
	std::remove(datafile.c_str());

	// Fixme. Should ask reluplex to generate at least one sample for each class.
	//std::fstream out(datafile, std::fstream::out);
	//out << -1.70586e-01 << "," << 5.38879e-02 << "," 
	//	<< -6.72141e-03 << "," << -3.10080e-02  << ",";
	//out << "Left\n";
	//out << -3.28355e-01 << "," << -1.09655e-01 << "," 
	//	<< -3.96630e-03 << "," << 3.00918e-02 << ",";
	//out << "Stay\n";
	//out << -4.21625e-01 << "," << -2.76742e-01 << "," 
	//	<< -1.65369e-02 << "," << 1.37284e-01 << ",";
	//out << "Right\n";		
	//out.flush ();
	//out.close ();

	agentSample (graph, 3, keyset);

	// Iterative buidling an abstraction of a neuron controller.
	do {
		// Execute the abstraction 
		Pole game;
		game.reset();
		game.perturbation();

		int size = keyset.size();
		std::cout << "before ce-gen keyset.size() == " << keyset.size() << "\n";
		bool res = ce_gen (game, graph, dt, 200, keyset, false);
		std::cout << "after  ce-gen keyset.size() == " << keyset.size() << "\n";
		if (!res) {
			// We found a counterexample to the neural net.
			std::cout << "A real counterexample is found!\n";
			// Should terminate reporting the counterexample.
		} else {
			if (keyset.size() == size) {
				g++;
			} else {
				g = 0;
				std::fstream evl(datafile, std::fstream::out);

				for (std::vector<std::pair<polestate,int>>::iterator i = keyset.begin(); 
						i != keyset.end(); ++i) { 
					render (evl, i->first, i->second);
					evl.flush ();
				}
				evl.close();
				if (keyset.size() > 200) {
					std::cout << "keyset.size() == " << keyset.size() << "\n";
					dt.learn(datafile);
				}
			}
		}
	} while (g < 10); // Verification converges after consecutive success in several rounds.
	std::cout << "Verifed!\n";

	int ts = 0;
	for(int g = 0; g < 100; ++g)
	{
		Pole game;
		game.reset();
		game.perturbation();
		display(game.getState());
		int s = 0;
		while(true)
		{
			polestate st = game.getState();
			double*  dtst = new double[4];
			dtst[0] = st.get<0>();dtst[1] = st.get<1>();dtst[2] = st.get<2>();dtst[3] = st.get<3>();
			int ac = dt.predict (dtst, 4);
			delete [] dtst;
			game.step(ac-1);
			
			bool failed = game.fail();
			s ++;
			ts ++;
			if( failed ) {
				std::cout << "Maintained in " << s << " steps\n";
				break;
			}
		}
	}
	std::cout << (ts / 100.0) << "\n";
} 