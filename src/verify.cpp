// Main verification engine of neural networks
//===================  Test and Verification of learned neural model =================== 
#include "qlearner/action.h"
#include "qlearner/qlearner.hpp"
#include "qlearner/stats.h"
#include <climits>
#include <ctype.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <stdlib.h>

#include "config.h"
#include "net/fc_layer.hpp"
#include "net/relu_layer.hpp"
#include "net/tanh_layer.hpp"
#include "net/solver.hpp"
#include "net/rmsprop.hpp"
#include "net/network.hpp"

#include "dt.h"
#include "Game.h"
#include "verify.h"
#include "safe-monitor.h"


#include "pole.h"
#include "pong.h"
#include "thermostat.h"
#include "drive.h"

// Used to define the libalf name space.
#include <libalf/alf.h>
// Angluin's L* algorithm
#include <libalf/algorithm_angluin.h>

using namespace net;
using namespace qlearn;
using namespace libalf;

int numSimulations;

int python_mutex = 0;

// Collect samples for each prediction class.
void agentSample (Game& game, ComputationGraph& graph, int acts, std::vector<std::pair<gamestate,int>>& keyset) {
	game.reset();
	game.perturbation();
	int s = 0;
	int th = 0;
	int c = 0;
	while (1) {
		gamestate st = game.getGameState();
		auto ac = getAction(graph, data(st));
		if (ac.id == th) {
			keyset.push_back(std::make_pair (st, ac.id));
			th++;
			c++;
		}
		game.step(ac.id);
		if (c == acts) break;
		bool failed = game.fail();
		bool terminated = game.terminate();
		s++;
		if( failed || terminated) {
			game.reset();
			game.perturbation();
		}
	}
}

// test a learn model in file.
void agentPlay (Game& game, std::string filename, int times, int bound=INT_MAX) {
	Network agent;
	agent.load (filename);
	ComputationGraph graph(agent);
	float reward = 0;
	int ts = 0;

	int succ = 0;
	for(int g = 0; g < times; ++g)
	{
		game.reset();
		game.perturbation();
		display(game.getGameState());
		int s = 0;
		bool terminated = false;
		while(s < bound)
		{
			auto ac = getAction(graph, data(game.getGameState()));
			std::cout << s << ": "; display (game.getGameState(), ac.id);
			game.step(ac.id);
			float currReward = 0;
			bool failed = game.fail();
			terminated = game.terminate();
			if (failed) currReward = -1;
			reward += currReward;
			ts ++;
			s ++;
			if( failed || terminated ) {
				std::cout << "Played in " << s << " steps\n";
				break;
			}
		}
		if (s == bound || terminated) succ++;
	}
	std::cout << (ts / times) << "\n";
	std::cout << succ << " out of " << times << " terminates safely.\n";
}

// test a learned model under a certain bound.
bool agentPlay (Game& game, ComputationGraph& graph, int bound) {
	int s = 0;
	bool terminated = false;
	while (s < bound) {
		auto ac = getAction(graph, data(game.getGameState()));
		game.step(ac.id);
		bool failed = game.fail();
		terminated = game.terminate();
		s++;
		if( failed || terminated ) break;
	}
	//std::cout << "agent played " << s << " step\n";
	return (s == bound || terminated);
}

bool agentPlay (Game& game, DT& dt, int bound) {
	int s = 0;
	bool terminated = false;
	while (s < bound) {
		gamestate st = game.getGameState();
		double*  dtst = new double[game.inputs()];
		for (int i = 0; i < game.inputs(); i++) {
			dtst[i] = st[i];
		}
		int ac = dt.predict (dtst, game.inputs());
		delete [] dtst;
		game.step(ac);
		bool failed = game.fail();
		terminated = game.terminate();
		s++;
		if( failed || terminated ) break;
	}
	//std::cout << "agent played " << s << " step\n";
	return (s == bound || terminated);
}

// export a learned model
void networkToJson (std::string filename) {
	Network agent;
	agent.load (filename);
	agent.saveJson (filename + ".json");
}

// UCT advisor
int uct_advise (UCTGameSimulator& sim2, gamestate st, bool terminal) {
    UCT::UCTPlanner uct (&sim2, -1, 1000, 1, 0.95);

    UCTGameState* current = new UCTGameState (st);
    uct.setRootNode (current, sim2.getActions(), 0, terminal);
    uct.plan();
    numSimulations += uct.numSimulations;
    UCT::SimAction* action = uct.getAction();
    const UCTGameAction* act = dynamic_cast<const UCTGameAction*> (action);
    return (act->id);
}

// Train NN with key points
bool key_gen (Game& game, UCTGameSimulator& sim, ComputationGraph& graph, int bound, std::vector<std::pair<gamestate,int>>& keyset) {
	if (bound == 0)
		return true;

	// game should be in the very initial state.
	game.reset();
	game.perturbation();

	std::vector<std::pair<gamestate,int>> path;
	int s = 0;
	bool terminated = false;
	while(s < bound)
	{
		//boost::tuples::tuple<double, double, double, double> 
		gamestate st = game.getUCTState();
		auto ac = getAction(graph, data(game.getGameState()));
		path.push_back (std::make_pair(st, ac.id));
		game.step(ac.id);
		display (st, ac.id);
		bool failed = game.fail();
		terminated = game.terminate();
		s ++;
		if( failed || terminated) break;
	}
	std::cout << "how well is the trained network: " << s << " steps\n";
	std::cout << "Trained network terminated? : " << terminated << "\n"; 
	if (s == bound || terminated) {
		// neural network is good enough so there is no need to improve it.
		return true;
	}
	else {
		sim.reset(&graph);
		for (std::vector<std::pair<gamestate,int>>::reverse_iterator i = path.rbegin(); i != path.rend(); ++i) { 
			// Check how mc thinks about the best move.
			//std::cout << "from the " << s << " step in the counterexample\n";
			game.setUCTState(i->first);
			int ac = uct_advise(sim, i->first, game.fail()||game.terminate());
			if (ac == i->second) {
				// The mc and neural net agrees with each other.
			} else {
				std::vector<std::pair<gamestate,int>> local_keyset;
				gamestate st = i->first;
				game.setUCTState (st);
				int i = 1;
				bool res = false;
				while (i <= bound - s) {
					local_keyset.push_back (std::make_pair (game.getGameState(), ac));
					game.step(ac);
					if (game.fail()) break;
					if (game.terminate()) { res = true; break; }
					st = game.getUCTState();
					//std::cout << "agent needs to play " << bound - s - i << " steps from "; display (st);
					res = agentPlay (game, graph, bound - s - i);
					game.setUCTState(st);
					if (res) break;
					ac = uct_advise(sim, st, game.fail()||game.terminate());
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

// Internally used only
void cegis_train(Game& game, UCTGameSimulator& sim, int bound, Network& network, ComputationGraph& graph) {
	std::vector<std::pair<gamestate,int>> keyset;

	int g = 0;
	numSimulations = 0;
	do {
		int size = keyset.size();
		bool fixable = key_gen (game, sim, graph, bound, keyset);
		if (!fixable) {
			//if (game.inv_mode) {
			//	std::cout << "Cannot fix an trace in the invaraint-based learning settting\n";
			//	exit (EXIT_FAILURE);
			//}
			std::cout << "Ignoring an unfixable trace...\n";
			g = 0;
			continue;
		}
		if (keyset.size() == size) {
			g++;
		} else {
			std::cout << "Current model has been good in " << g << " times consecutively.\n";
			g = 0;
			// Supervised learning to train the controller.
			auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.005, 0.001));
			RMSProp* rmsprop = prop.get();
			Solver solver( std::move(prop) );
			// First learn on the new keypoints
			int error;
			int currSize = keyset.size() - size;
			std::cout << "---------------------------------------------------------------\n";
			std::cout << "Training on " << currSize << " samples\n";
			int iteri = 0;
			do {
				error = 0;
				int index = 0;	
				for (std::vector<std::pair<gamestate,int>>::reverse_iterator i = keyset.rbegin(); 
						i != keyset.rend() && index < currSize; ++i) {
					//std::cout << "Enforcing the neural network to behave as: \n";
					//display (i->first, i->second);

					//float target_value = 1.0;
			
					const auto& result = graph.forward(data(i->first));
					int row, col;
					result.maxCoeff(&row,&col);
					error += (row == i->second? 0 : 1);
					//std::cout << "Its original behavior on this is: " << row << "\n";
					//for (int k = 0; k < result.size(); k ++)
					//	std::cout << result[k] << " ";
					//std::cout << "\n";
					if (row != i->second) {
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
				for (std::vector<std::pair<gamestate,int>>::iterator i = keyset.begin(); 
						i != keyset.end(); ++i) { 
					
					//float target_value = 1.0;
			
					const auto& result = graph.forward(data(i->first));
					int row, col;
					result.maxCoeff(&row,&col);
					error += (row == i->second? 0 : 1);

					if (row != i->second || game.actions() <= 2) { // binary decision can easily lead to overfitting ...
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
				}
				network.update( solver );
				iteri++;
				//std::cout << "error number: " << error << "\n";
			} while (error > 0 && iteri < 10000);
			std::cout << "batch training error number: " << error << "\n";
		}
	} while (g < 2000);
	std::cout << "Trained Successfully!\n";
	std::cout << "Number of MCTS simulations: " << numSimulations << "\n";
}

// CEGIS based training for controller synthesis based on an existing network model.
void cegis_train(Game& game, UCTGameSimulator& sim, int bound, std::string model) {
	Network network;
	network.load(model);
	ComputationGraph graph(network);

	cegis_train(game, sim, bound, network, graph);

	// Assign a new name to the model
	int i = model.size(), suffix = 0;
	std::string modelname;
	for(string::reverse_iterator k = model.rbegin(); k != model.rend(); ++k) {
    	if (isdigit(*k)) 
    		i--;
	}
	if (i != model.size()) {
		suffix = std::stoi(model.substr(i));
		suffix ++;
		modelname = model.substr(0, i);
	} else {
		suffix ++;
		modelname = model.substr(0, model.find("."));
	}

	std::string new_model_name = modelname + std::to_string(suffix) + ".network";
	network.save (new_model_name);

	agentPlay(game, new_model_name, 1000, bound);
}

// CEGIS based training for controller synthesis.
void cegis_train(Game& game, UCTGameSimulator& sim, int bound) {
	Network network;
	network << FcLayer(Matrix::Random(10, game.inputs()).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(10, 10).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(game.actions(), 10).array() / 5);
	network << TanhLayer(Matrix::Zero(game.actions(), 1));
	
	ComputationGraph graph(network);

	cegis_train(game, sim, bound, network, graph);

	network.save ("supervised_agent.network");

	agentPlay(game, "supervised_agent.network", 1000, bound);
}

bool key_gen (Game& game, UCTGameSimulator& sim, DT& dt, int bound, std::vector<std::pair<gamestate,int>>& keyset) {
	if (bound == 0)
		return true;

	// game should be in the very initial state.
	game.reset();
	game.perturbation();

	std::vector<std::pair<gamestate,int>> path;
	int s = 0;
	bool terminated = false;
	while(s < bound)
	{
		//boost::tuples::tuple<double, double, double, double> 
		gamestate st = game.getUCTState();
		double*  dtst = new double[game.inputs()];
		for (int i = 0; i < game.inputs(); i++) {
			dtst[i] = st[i];
		}
		int ac = dt.predict (dtst, game.inputs());
		delete [] dtst;
		path.push_back (std::make_pair(st, ac));
		game.step(ac);
		display (st, ac);
		bool failed = game.fail();
		terminated = game.terminate();
		s ++;
		if( failed || terminated) break;
	}
	std::cout << "how well is the trained network: " << s << " steps\n";
	std::cout << "Trained network terminated? : " << terminated << "\n"; 
	if (s == bound || terminated) {
		// neural network is good enough so there is no need to improve it.
		return true;
	}
	else {
		sim.reset(&dt);
		for (std::vector<std::pair<gamestate,int>>::reverse_iterator i = path.rbegin(); i != path.rend(); ++i) { 
			// Check how mc thinks about the best move.
			//std::cout << "from the " << s << " step in the counterexample\n";
			game.setUCTState(i->first);
			int ac = uct_advise(sim, i->first, game.fail()||game.terminate());
			if (ac == i->second) {
				// The mc and neural net agrees with each other.
			} else {
				std::vector<std::pair<gamestate,int>> local_keyset;
				gamestate st = i->first;
				game.setUCTState (st);
				int i = 1;
				bool res = false;
				while (i <= bound - s) {
					local_keyset.push_back (std::make_pair (game.getGameState(), ac));
					game.step(ac);
					if (game.fail()) break;
					if (game.terminate()) { res = true; break; }
					st = game.getUCTState();
					//std::cout << "agent needs to play " << bound - s - i << " steps from "; display (st);
					res = agentPlay (game, dt, bound - s - i);
					game.setUCTState(st);
					if (res) break;
					ac = uct_advise(sim, st, game.fail()||game.terminate());
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

void cegis_trainDT(Game& game, UCTGameSimulator& sim, int bound, DT& dt, int dtSize) {
	std::vector<std::pair<gamestate,int>> keyset;

	int g = 0;
	numSimulations = 0;

	std::string datafile = "supervised_dt_agent.data";
	std::remove(datafile.c_str());

	UCT::UCTPlanner uct (&sim, -1, 1000, 1, 0.95);
	// Sample one sample for one action
	game.reset();
	game.perturbation();
	int s = 0;
	int th = 0;
	int c = 0;
	double r = 0;
	while (1) {
		gamestate st = game.getGameState();
		UCTGameState* ugs = new UCTGameState (game.getUCTState());
		uct.setRootNode(ugs, sim.getActions(), r, game.terminate() || game.fail());
		delete ugs;
        uct.plan();
        UCT::SimAction* action = uct.getAction();
        const UCTGameAction* act = dynamic_cast<const UCTGameAction*> (action);

		if (act->id == th) {
			keyset.push_back(std::make_pair (st, act->id));
			th++;
			c++;
		}
		game.step(act->id);
		if (c == game.actions()) break;
		bool failed = game.fail();
		bool terminated = game.terminate();
		s++;
		if( failed || terminated) {
			game.reset();
			game.perturbation();
			r = 0;
		}
		r = 1;
	}
	// Supervised Learning of a DT controller
	do {
		int size = keyset.size();
		bool fixable = key_gen (game, sim, dt, bound, keyset);
		if (!fixable) {
			//if (game.inv_mode) {
			//	std::cout << "Cannot fix an trace in the invaraint-based learning settting\n";
			//	exit (EXIT_FAILURE);
			//}
			std::cout << "Ignoring an unfixable trace...\n";
			g = 0;
			continue;
		}
		if (keyset.size() == size) {
			g++;
		} else {
			std::cout << "Current model has been good in " << g << " times consecutively.\n";
			g = 0;
			
			// Output trainning samples to file for a DT to learn.
			std::fstream evl(datafile, std::fstream::out);

			for (std::vector<std::pair<gamestate,int>>::iterator i = keyset.begin(); 
					i != keyset.end(); ++i) { 
				render (evl, i->first, i->second);
				evl.flush ();
			}
			evl.close();
			if (keyset.size() > 200) {
				std::cout << "keyset.size() == " << keyset.size() << "\n";
				dt.learn(datafile, game.inputs(), dtSize);
			}
		}
	} while (g < 2000);
	std::cout << "Trained Successfully!\n";
	std::cout << "Number of MCTS simulations: " << numSimulations << "\n";
}

// CEGIS based training for controller synthesis using decision tree.
void cegis_trainDT(Game& game, UCTGameSimulator& sim, int bound, int dtSize) {
	std::string modelname = "supervised_dt_agent.json";
	DT dt(modelname);
	// Learning a DT
	cegis_trainDT(game, sim, bound, dt, dtSize);
	// Testing the DT
	dt.interprete("supervised_dt_agent.abstraction", game.inputs());
	std::cout << "Testing learned abstraction:\n";
	int ts = 0;
	int succ = 0;
	for(int g = 0; g < 1000; ++g)
	{
		game.reset();
		game.perturbation();
		display(game.getGameState());
		int s = 0;
		bool terminated = false;
		while(s < bound)
		{
			gamestate st = game.getGameState();
			double*  dtst = new double[game.inputs()];
			for (int i = 0; i < game.inputs(); i++) {
				dtst[i] = st[i];
			}
			int ac = dt.predict (dtst, game.inputs());
			delete [] dtst;
			game.step(ac);
			
			bool failed = game.fail();
			terminated = game.terminate();
			s ++;
			ts ++;
			if( failed || terminated) {
				std::cout << "Maintained in " << s << " steps\n";
				break;
			}
		}
		if (s >= bound || terminated) 
			succ++;
	}
	std::cout << (ts / 1000.0) << "\n";
	std::cout << succ << " out of 1000 games is successful.\n";
}


// CEGIS based training for controller synthesis using linear regression.

// Verify an abstraction
bool ce_gen (Game& game, ComputationGraph& graph, DT& dt, int bound, std::vector<std::pair<gamestate,int>>& keyset, bool rec_flag) {
	if (bound == 0)
		return true;
	std::vector<std::pair<gamestate,int>> path;
	int s = 0;
	bool terminated = false;
	while(s < bound)
	{
		//boost::tuples::tuple<double, double, double, double> 
		gamestate st = game.getGameState();
		double*  dtst = new double[game.inputs()];
		for (int i = 0; i < game.inputs(); i++) {
			dtst[i] = st[i];
		}
		int ac = dt.predict (dtst, game.inputs());
		delete [] dtst;
		path.push_back (std::make_pair(game.getUCTState(), ac));
		game.step(ac);
			
		bool failed = game.fail();
		terminated = game.terminate();
		s ++;
		if( failed || terminated ) break;
	}
	if (!rec_flag) {
		std::cout << "how well is the abstraction: " << s << " steps\n";
		std::cout << "Learned abstraction terminated? : " << terminated << "\n";
	}
	if (s == bound || terminated) {
		// abstraction is fine so there is no counterexample
		return true;
	}
	else {
		for (std::vector<std::pair<gamestate,int>>::reverse_iterator i = path.rbegin(); i != path.rend(); ++i) { 
			game.setUCTState (i->first);
			// Check how neural net works.
			auto ac = getAction(graph, data(game.getGameState()));
			if (ac.id == i->second) {
				// The abstraction and neural net agrees with each other.
			} else {
				// Check if neural net's decision is good.
				bool res = agentPlay (game, graph, bound - s);
				game.setUCTState(i->first);
				if (res) {
					keyset.push_back (std::make_pair (game.getGameState(), ac.id));
					game.step(ac.id);
					return ce_gen (game, graph, dt, bound - s - 1, keyset, true);
				}
			}
			s--;
		}
		return false;
	}
}

// abstract-refine based verification of a neural model stored in filename.
// with the goal of checking wether the agent can behave safely within bounded steps.
void abstraction_check (Game& game, std::string filename, int bound, int dtSize) {
	Network agent;
	agent.load (filename);
	std::string modelname = filename + ".json";
	agent.saveJson (modelname);
	ComputationGraph graph(agent);
	int g = 0;

	// Use a decision tree to represent the abstraction of the nueral net
	DT dt(modelname);
	std::vector<std::pair<gamestate,int>> keyset;
	// Use keyset to refine the abstraction
	std::string datafile = filename+".data";
	std::remove(datafile.c_str());

	agentSample (game, graph, game.actions(), keyset);

	// Iterative buidling an abstraction of a neuron controller.
	do {
		// Execute the abstraction 
		game.reset();
		game.perturbation();

		int size = keyset.size();
		std::cout << "before ce-gen keyset.size() == " << keyset.size() << "\n";
		bool res = ce_gen (game, graph, dt, bound, keyset, false);
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

				for (std::vector<std::pair<gamestate,int>>::iterator i = keyset.begin(); 
						i != keyset.end(); ++i) { 
					render (evl, i->first, i->second);
					evl.flush ();
				}
				evl.close();
				//if (keyset.size() > 200) {
					//std::cout << "keyset.size() == " << keyset.size() << "\n";
					dt.learn(datafile, game.inputs(), dtSize);
				//}
			}
		}
	} while (g < 50000); // Verification converges after consecutive success in several rounds.
	std::cout << "Verifed!\n";
	// Abstraction to file for hybrid system verification!
	dt.interprete(filename+".abstraction", game.inputs());
	std::cout << "Testing learned abstraction:\n";
	int ts = 0;
	int succ = 0;
	for(int g = 0; g < 10000; ++g)
	{
		game.reset();
		game.perturbation();
		//display(game.getGameState());
		gamestate initial = game.getGameState();
		int s = 0;
		bool terminated = false;
		while(s < bound)
		{
			gamestate st = game.getGameState();
			double*  dtst = new double[game.inputs()];
			for (int i = 0; i < game.inputs(); i++) {
				dtst[i] = st[i];
			}
			int ac = dt.predict (dtst, game.inputs());
			delete [] dtst;
			game.step(ac);
			
			bool failed = game.fail();
			terminated = game.terminate();
			s ++;
			ts ++;
			if( failed || terminated) {
				display (initial);
				std::cout << "Only maintained in " << s << " steps\n";
				break;
			}
		}
		if (s >= bound || terminated) 
			succ++;
	}
	std::cout << (ts / 10000.0) << "\n";
	std::cout << succ << " out of 10000 games are successful.\n";
} 

// void testPolicyFile (std::string policy) {
// 	std::cout << "Testing a given policy file:\n";
// 	int ts = 0;
// 	int succ = 0;
// 	for(int g = 0; g < 1000; ++g)
// 	{
// 		game.reset();
// 		game.perturbation();
// 		display(game.getGameState());
// 		int s = 0;
// 		bool terminated = false;
// 		while(s < bound)
// 		{
// 			gamestate st = game.getGameState();
// 			double*  dtst = new double[game.inputs()];
// 			for (int i = 0; i < game.inputs(); i++) {
// 				dtst[i] = st[i];
// 			}
// 			int ac = dt.predict (dtst, game.inputs());
// 			delete [] dtst;
// 			game.step(ac);
			
// 			bool failed = game.fail();
// 			terminated = game.terminate();
// 			s ++;
// 			ts ++;
// 			if( failed || terminated) {
// 				std::cout << "Maintained in " << s << " steps\n";
// 				break;
// 			}
// 		}
// 		if (s >= bound || terminated) 
// 			succ++;
// 	}
// 	std::cout << (ts / 1000.0) << "\n";
// 	std::cout << succ << " out of 1000 games is successful.\n";
// }

/*
void testNetworkTraining () {
	Network network;
	network << FcLayer(Matrix::Random(10, 5).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(10, 10).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(3, 10).array() / 5);
	network << TanhLayer(Matrix::Zero(3, 1));
	
	ComputationGraph graph(network);

	Vector vec;
	vec.resize(5);
	vec[0] = 0.7;
	vec[1] = 0.195;
	vec[2] = 0.82;
	vec[3] = 1;
	vec[4] = -0.55;

	auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.0005, 0.001));
	RMSProp* rmsprop = prop.get();
	Solver solver( std::move(prop) );
	int error;
	do {
		error = 0;
		const auto& result = graph.forward(vec);
		int row, col;
		result.maxCoeff(&row,&col);
		error += (row == 2? 0 : 1);
		std::cout << "Its original behavior on this is: " << row  << "\n";
		for (int k = 0; k < result.size(); k ++)
			std::cout << result[k] << " ";
		std::cout << "\n";
		Vector errorCache = Vector::Zero( result.size() );

		//float delta = result[i->second] - target_value;
		//errorCache[i->second] = delta;
		for (int k = 0; k < result.size(); k++) {
			if (k == 2) 
				errorCache[k] = result[k] - 1.0;
			else
				errorCache[k] = result[k] - (-1.0);
		}

		graph.backpropagate(errorCache, solver);
		network.update(solver);
	} while (error > 0);

}

void answer_Membership (std::list<int> query, std::list<std::list<int>>& experiences) {
	// Should ask a hybrid system for membership query
	// Fixme: but use existing experiences to answer for now.
}

bool check_Equivalence (conjecture * cj, std::list<std::list<int>>& experiences, 
						Game& game, Dt& dt, std::list<int>& ce, double err, double conf) {
	// Sample a sufficient number of example inputs.
	int sample_size = 0;

	for(int g = 0; g < sample_size; ++g)
	{
		game.reset();
		game.perturbation();
		display(game.getGameState());
		int s = 0;
		bool terminated = false;
		while(true)
		{
			gamestate st = game.getGameState();
			double*  dtst = new double[game.inputs()];
			for (int i = 0; i < game.inputs(); i++) {
				dtst[i] = st[i];
			}
			int ac = dt.predict (dtst, game.inputs());
			delete [] dtst;
			game.step(ac);
			
			bool failed = game.fail();
			terminated = game.terminate();
			s ++;
			if( failed || terminated) {
				break;
			}
		}
		if (s >= bound || terminated) {// Good case!
		} else {
			return false;
		}
	}
	return true;
}

// Use pac learning to verify a neural agent controller.
// Idea: Compute a DFA to approximate all the possible decsion vectors of an abstaction.
void pac_verify (Game& game, std::string filename, int bound, int n_jumps, double err, double conf) {
	Network agent;
	agent.load (filename);
	ComputationGraph graph(agent);

	int alphabet_size = n_jumps;
	knowledgebase<bool> base;
	// Create learning algorithm (Angluin L*) without a logger (2nd argument is NULL)
	angluin_simple_table<bool> algorithm(&base, NULL, alphabet_size);
	conjecture * result = NULL;

	do {
		conjecture * cj = algorithm.advance();
		if (cj == NULL) {
			list<list<int> > queries = base.get_queries();
			list<list<int> >::iterator li;
			for (li = queries.begin(); li != queries.end(); li++) {
				bool a = answer_Membership(*li);
				base.add_knowledge(*li, a);
			}
		} else {
			list<int> ce;
			bool is_equivalent = check_Equivalence(cj, ce, err, conf);
			if (is_equivalent) {
				result = cj;
			} else {
				algorithm.add_counterexample(ce);
				delete cj;
			}
		}
	} while (result == NULL);

	std::cout << endl << "Sucessfully verified with:" << endl << result->visualize() << endl;
	delete result;	
}
*/
void train (Game& game, Game& test);

// =======================  Correct by construction machine learning ======================= 
typedef std::unordered_map<int, std::pair<double,double>> partial_states; // conjunction of ranges
typedef std::vector<partial_states> unsafes;  // disjunction of conjunctions of ranges
void correct_by_construction_train (Game& game, 
		std::string agent,
		std::vector<std::pair<double,double>>& initset,
		unsafes& unsafeset,
		std::vector<std::pair<double, double>>& stateset);

int main(int argc, char** argv)
{
	if (argc > 1) {
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		std::string param(argv[1]);
		if (param.compare ("uct") == 0) {
			std::string gamename(argv[2]);

			UCTGameSimulator* sim; 
		    UCTGameSimulator* sim2;
		    if (gamename.compare ("pong") == 0) {
		    	sim = new UCTPongSimulator();
		    	sim2 = new UCTPongSimulator();
		    } else if (gamename.compare ("pole") == 0) {
		    	sim = new UCTPoleSimulator ();
		    	sim2 = new UCTPoleSimulator ();
		    } else if (gamename.compare ("thermostat") == 0) {
		    	sim = new UCTThermostatSimulator ();
		    	sim2 = new UCTThermostatSimulator ();
		    } else if (gamename.compare ("drive") == 0) {
		    	sim = new UCTDriveSimulator();
		    	sim2 = new UCTDriveSimulator();
		    } else {
		    	std::cout << "The game " << gamename << "is not found.\n";
				return -1;
		    }

		    std::vector<double> init;
		    for (int i = 3; i < argc; i++) {
		    	std::string value(argv[i]);
		    	init.push_back(std::stod(value));
		    }
		    if (!init.empty()) {
		    	UCTGameState s (init);
		    	sim->setState(&s);
		    }

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
		        cout << "Game:" << i << "  steps: " << steps << "  r: " << r << endl;
		        cout << "Game terminate ? " << sim->game->terminate() << endl;
		        sim->reset();
			}
			delete sim2;
			delete sim;
		} 
		else if (param.compare ("cegis") == 0) {
			std::string gamename(argv[2]);
			std::string modelname = "";
			if (argc > 3) {
				std::string s(argv[3]);
				modelname += s;
			}	

			if (gamename.compare ("pong") == 0) {
				Pong game;
				UCTPongSimulator sim;
				if (modelname.empty()) cegis_train(game, sim, INT_MAX);
				else cegis_train(game, sim, INT_MAX, modelname);
			} else if (gamename.compare ("pole") == 0) {
				Pole game;
				UCTPoleSimulator sim;
				if (modelname.empty()) cegis_train(game, sim, 200);
				else cegis_train(game, sim, 200, modelname);
			} else if (gamename.compare ("thermostat") == 0) {
				Thermostat game;
				UCTThermostatSimulator sim;
				if (modelname.empty()) cegis_train (game, sim, 200);
				else cegis_train(game, sim, 200, modelname);
			} else if (gamename.compare ("drive") == 0) {
				Drive game;
				UCTDriveSimulator sim;
				if (modelname.empty()) cegis_train (game, sim, 200);
				else cegis_train(game, sim, 200, modelname); 
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			}
		}
		else if (param.compare("cegis-inv") == 0) {
			std::string gamename(argv[2]);
			if (gamename.compare ("pole") == 0) {
				Pole game;
				UCTPoleSimulator sim;
				game.inv_mode = true;
				sim.pole_.inv_mode = true;
				cegis_train(game, sim, 15);
			} 
			else if (gamename.compare("thermostat") == 0) {
				Thermostat game;
				UCTThermostatSimulator sim;
				game.inv_mode = true;
				sim.t_.inv_mode = true;
				cegis_train (game, sim, 15);
			}
			else if (gamename.compare("drive") == 0) {
				Drive game;
				UCTDriveSimulator sim;
				game.inv_mode = true;
				sim.t_.inv_mode = true;
				cegis_train (game, sim, 15);
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			}
		}
		else if (param.compare("play") == 0) {
			std::string gamename(argv[2]);
			std::string model(argv[3]);
			int bound = INT_MAX;
			int times = 100;
			if (argc > 4) {
				std::string boundstr(argv[4]);
				bound = std::stoi(boundstr);
			}
			if (argc > 5) {
				std::string timestr(argv[5]);
				times = std::stoi(timestr);
			}
			if (gamename.compare ("pong") == 0) {
				Pong game;
				agentPlay(game, model, times, bound);
			} else if (gamename.compare ("pole") == 0) {
				Pole game;
				agentPlay(game, model, times, bound);
			} else if (gamename.compare ("thermostat") == 0) {
				Thermostat game;
				agentPlay (game, model, times, bound);
			} else if (gamename.compare ("drive") == 0) {
				Drive game;
				agentPlay (game, model, times, bound);
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			} 
		} 
		else if (param.compare("verify") == 0){
			//Network agent;
			//agent.load (agentfile);
			//ComputationGraph graph(agent);

			//vec.resize (4);
			//vec[0] = -0.189315;
			//vec[1] = -0.00291353;
			//vec[2] = 0.00289438;
			//vec[3] = 0.00875231;


			//auto ac = getAction(graph, vec);
			//std::cout << "decision : " << ac.id << "\n";

			//agentPlay (param);
			//networkToJson(agentfile);
			
			// -- doing abstraction refinement --
			std::string gamename(argv[2]);
			std::string model(argv[3]);
			int bmc_bound = 200;
			if (argc > 4) {
				std::string dt_size_str(argv[4]);
				bmc_bound = std::stoi(dt_size_str);
			}
			int dtSize = 31;
			if (argc > 5) {
				std::string dt_size_str(argv[5]);
				dtSize = std::stoi(dt_size_str);
			}
			if (gamename.compare("pong") == 0) {
				Pong game;
				abstraction_check (game, model, INT_MAX, dtSize);
			} else if (gamename.compare("pole") == 0) {
				Pole game;
				abstraction_check (game, model, bmc_bound, dtSize);
			} else if (gamename.compare("thermostat") == 0) {
				Thermostat game;
				abstraction_check (game, model, bmc_bound, dtSize);
			} else if (gamename.compare ("drive") == 0) {
				Drive game;
				abstraction_check (game, model, bmc_bound, dtSize);
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			}
		}
		else if (param.compare ("verify-inv") == 0) {
			std::string gamename(argv[2]);
			std::string model(argv[3]);
			int dtSize = 31;
			if (argc > 4) {
				std::string dt_size_str(argv[4]);
				dtSize = std::stoi(dt_size_str);
			}
			if (gamename.compare ("thermostat") == 0) {
				Thermostat game;
				game.inv_mode = true;
				abstraction_check (game, model, 15, dtSize);
			}
			else if (gamename.compare ("drive") == 0) {
				Drive game;
				game.inv_mode = true;
				abstraction_check (game, model, 15, dtSize);
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			}
		}
		else if (param.compare("train") == 0) {
			std::string gamename(argv[2]);
			if (gamename.compare("pong") == 0) {
				Pong game;
				Pong test;
				train (game, test);
			} else if (gamename.compare("pole") == 0) {
				Pole game;
				Pole test;
				train (game, test);
			} else if (gamename.compare("thermostat") == 0) {
				Thermostat game;
				Thermostat test;
				train (game, test);
			} else if (gamename.compare ("drive") == 0) {
				Drive game;
				Drive test;
				train (game, test);
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			}
		} 
		else if (param.compare("correct-construct") == 0) {
			std::string gamename(argv[2]);
			std::string agent(argv[3]);
			if (gamename.compare ("drive") == 0) {
				Drive game;
				std::vector<std::pair<double,double>> initset;
				unsafes unsafeset;
				std::vector<std::pair<double, double>> stateset;

				initset.push_back (std::make_pair(-1,1));
				initset.push_back (std::make_pair(-0.78539815,0.78539815));

				partial_states e1;
				e1[0] = std::make_pair(-2.2, -2.0);
				unsafeset.push_back(e1);
				partial_states e2;
				e2[0] = std::make_pair(2.0, 2.2);
				unsafeset.push_back(e2);

				stateset.push_back (std::make_pair(-2.2,2.2));
				stateset.push_back (std::make_pair(-1.6,1.6));

				correct_by_construction_train (game, agent, initset, unsafeset, stateset);
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			}
		}
		else if (param.compare("safe-monitor") == 0) {
			std::string gamename(argv[2]);
			std::string agent(argv[3]);   // controller learned as a neural network for which a (verfied) safe monitor will be learned.
			std::string safeset(argv[4]); // Presumed safe set learned and approximiated as a neural network.
			int n_samples = 1000;         // number of samples used to check DT approaximation correctness at a time.
			int n_bootstrap = 200;		  // number of samples used to bootstrap the DT approximation of a set state set
			int n_stops = 1;
			if (argc > 5) {
				std::string nstr(argv[5]);
				n_samples = std::stoi(nstr);
			}
			if (argc > 6) {
				std::string bstr(argv[6]);
				n_bootstrap = std::stoi(bstr);
			}
			if (argc > 7) {
				std::string sstr(argv[7]);
				n_stops = std::stoi(sstr);
			}
			if (gamename.compare ("drive") == 0) {
				Drive game;
				std::vector<std::pair<double, double>> stateset;
				stateset.push_back (std::make_pair(-2.2,2.2));
				stateset.push_back (std::make_pair(-1.6,1.6));
				train_monitor (game, agent, safeset, stateset, n_samples, n_bootstrap, n_stops);
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			}
		}

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    	cout << "Executed in " << duration << " microseconds.\n";
		return 0;
	}
	return 0;
}

std::mutex mTargetNet;
std::atomic<bool> evaluate(false);
std::atomic<bool> run(true);

void learn_thread( Game& game, Network& target_net, ComputationGraph& graph )
{
	Config config(game.inputs(), game.actions(), 2000000);
	config.epsilon_steps(2000000).update_interval(10000).batch_size(32).init_memory_size(10000).init_epsilon_time(100000)
		.discount_factor(0.98);
	
	Network network;
	network << FcLayer(Matrix::Random(10, game.inputs()).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(10, 10).array() / 5);
	network << ReLULayer(Matrix::Zero(10, 1));
	network << FcLayer(Matrix::Random(game.actions(), 10).array() / 5);
	//network << TanhLayer(Matrix::Zero(game.actions(), 1));
	
	qlearn::QLearner learner( config, std::move(network) );
	
	auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.0005, 0.001));
	RMSProp* rmsprop = prop.get();
	Solver solver( std::move(prop) );
	
	std::fstream rewf("reward.txt", std::fstream::out);

	game.reset ();
	game.perturbation ();

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

		game.step(ac);
		float r = 0;
		bool failed = game.fail();
		bool terminated = game.terminate();
		if (failed) r = -1;
		if (terminated) r = 1;

		ac = learner.learn_step( data (game.getGameState()), r, failed || terminated, solver );
		if( failed || terminated )
		{
			game.reset();
			game.perturbation ();
			games++;
		}
	}
}

void train (Game& game, Game& test) {
	Network network;
	ComputationGraph graph(network);
	std::thread learner( learn_thread, std::ref(game), std::ref(network), std::ref(graph) );
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
				test.reset();
				test.perturbation();
				for(int s = 0; s < 200; ++s)
				{
					auto ac = getAction(graph, data(test.getGameState()));
					test.step(ac.id);
					float currReward = 0;
					bool failed = test.fail();
					bool terminated = test.terminate();
					if (failed) currReward = -1;
					if (terminated) currReward = 1;
					reward += currReward;
					ts ++;
					if( failed || terminated) break;
				}
			}
			std::cout << "Steps: " << (ts / 200.0) << "\n";
			std::cout << "Rewards: " << reward << "\n";
			if (ts / 200.0 == 200.0 || reward == 200) {
				copy.save ("train_agent" + std::to_string(step) + ".network");
			}
			evl << reward << "\n";
			evl.flush();
			evaluate = false;
			step ++;
			sleep (10);
		}
	}
	run = false;
}

// =======================  Correct by construction machine learning ======================= agentplay
double getScore(ComputationGraph& critic, std::vector<double> sample) {
	const auto& result = critic.forward(data(sample));
	return result[0];
}

// Call nn agent to execute a step
std::vector<double> nn_step (Game& game, ComputationGraph& graph, std::vector<double> sample) {
	game.setGameState (sample);
	auto ac = getAction(graph, data(game.getGameState()));
	game.step(ac.id);
	return game.getGameState();
}

bool fixable(Game& game, std::vector<double> point, int& ac, ComputationGraph& critic) {
	double max = -1;
	for (int i = 0; i < game.actions(); i++) {
		game.setGameState(point);
		game.step(i);
		if (game.fail()) 
			continue;
		double s = getScore(critic, game.getGameState());
		if (s > max) {
			ac = i;
			max = s;
		}
	}
	return (max > 0);
}

bool agentEval (Game& game, ComputationGraph& actor, ComputationGraph& critic, int bound,
					std::vector<std::pair<std::vector<double>, double>>& counter_examples) 
{
	int s = 0;
	bool terminated = false;
	while (s < bound) {
		auto ac = getAction(actor, data(game.getGameState()));
		game.step(ac.id);
		bool failed = game.fail();
		terminated = game.terminate();
		s++;
		if( failed || terminated ) break;
		else {
			double score = getScore (critic, game.getGameState());
			if (score <= 0) 
				counter_examples.push_back (std::make_pair(game.getGameState(), 0.9));
		}
	}
	//std::cout << "agent played " << s << " step\n";
	return (s == bound || terminated);
}

bool agentEval (Game& game, ComputationGraph& actor, int bound, std::vector<std::vector<double>>& counter_examples) 
{
	int s = 0;
	bool terminated = false;
	while (s < bound) {
		auto ac = getAction(actor, data(game.getGameState()));
		game.step(ac.id);
		bool failed = game.fail();
		terminated = game.terminate();
		s++;
		if( failed || terminated ) break;
		else {
			counter_examples.push_back (game.getGameState());
		}
	}
	//std::cout << "agent played " << s << " step\n";
	return (s == bound || terminated);
}

// check pre condtions
bool sample_to_check_precondition (Game& game, ComputationGraph& actor, ComputationGraph& critic, 
							int n_samples, std::vector<std::pair<double, double>> initset,
							std::vector<std::pair<std::vector<double>, double>>& counter_examples, 
							std::vector<std::pair<std::vector<double>, int>>& train_examples) {
	int iteri = 0;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator (seed);
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);
	std::vector<std::vector<double>> points;
	while (iteri < n_samples) {
		// Sample a point within the range of intersted.
		std::vector<double> point;
		for (int j = 0; j < initset.size(); j++) {
			double start = initset[j].first;
			double end = initset[j].second;

			double sv =  (end - start) * uniform01(generator) + start;
			point.push_back(sv);
		}
		//std::cout << "Samping a value in range (";
		//	for (int j = 0; j < point.size(); j++) {
		//	std::cout << point[j];
		//}
		//std::cout << ")\n";
		points.push_back(point);
		iteri++;
	}
	bool verified = true;	
	for (int j = 0; j < points.size(); j++) {
		double score = getScore(critic, points[j]);
		//std::cout << "precondition sample score: " << score << "\n";
		if (score <= 0) {
			counter_examples.push_back(std::make_pair(points[j], 0.9));
			verified = false;
		}
	}
	return verified;
}

// check post condition
bool sample_to_check_postcondition (Game& game, ComputationGraph& actor, ComputationGraph& critic, 
							int n_samples, unsafes unsafeset, std::vector<std::pair<double, double>> stateset,
							std::vector<std::pair<std::vector<double>, double>>& counter_examples, 
							std::vector<std::pair<std::vector<double>, int>>& train_examples) {
	int iteri = 0;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator (seed);
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);
	std::vector<std::vector<double>> points;
	while (iteri < n_samples) {
		// Sample a point within the range of intersted.
		std::vector<double> point;
		// But firstly decide an unsafe region
		int i = rand() % unsafeset.size();
		for (int j = 0; j < game.inputs(); j++) {
			if (unsafeset[i].find(j) != unsafeset[i].end()) {
				double start = unsafeset[i][j].first;
				double end = unsafeset[i][j].second;

				double sv =  (end - start) * uniform01(generator) + start;
				point.push_back(sv);
			} else {
				double start = stateset[j].first;
				double end = stateset[j].second;

				double sv =  (end - start) * uniform01(generator) + start;
				point.push_back(sv);
			}
		}
		//std::cout << "Samping a value in range (";
		//	for (int j = 0; j < point.size(); j++) {
		//	std::cout << point[j];
		//}
		//std::cout << ")\n";
		points.push_back(point);
		iteri++;
	}
	bool verified = true;	
	for (int j = 0; j < points.size(); j++) {
		double score = getScore(critic, points[j]);
		//std::cout << "postcondition sample score: " << score << "\n";
		if (score > 0) {
			counter_examples.push_back(std::make_pair(points[j], -0.9));
			verified = false;
		}
	}
	return verified;
}

// check inductive invariant
bool sample_to_check_inv (Game& game, ComputationGraph& actor, ComputationGraph& critic, 
							int n_samples, std::vector<std::pair<double, double>> stateset,
							std::vector<std::pair<std::vector<double>, double>>& counter_examples, 
							std::vector<std::pair<std::vector<double>, int>>& train_examples) {
	int iteri = 0;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator (seed);
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);
	std::vector<std::vector<double>> points;
	while (iteri < n_samples) {
		// Sample a point within the range of intersted.
		std::vector<double> point;
		for (int j = 0; j < stateset.size(); j++) {
			double start = stateset[j].first;
			double end = stateset[j].second;

			double sv =  (end - start) * uniform01(generator) + start;
			point.push_back(sv);
		}
		//std::cout << "Samping a value in range (";
		//	for (int j = 0; j < point.size(); j++) {
		//	std::cout << point[j];
		//}
		//std::cout << ")\n";
		if (getScore(critic, point) > 0)
			points.push_back(point);
		iteri++;
	}
	std::cout << "Sampled " << points.size() << " instances in checking inductiveness.\n";
	bool verified = true;	
	for (int j = 0; j < points.size(); j++) {
		game.setGameState(points[j]);
		if (game.terminate()) {
			// Game is in the final state, which is safe, do nothing.
		} else if (game.fail()) {
			counter_examples.push_back(std::make_pair(points[j], -0.9));
			verified = false;
		} else {
			//std::vector<double> reach = nn_step (game, actor, points[j]);
			std::vector<std::pair<std::vector<double>, double>> ces;
			bool res = agentEval (game, actor, critic, 1000, ces);
			if (!res) {
				counter_examples.push_back(std::make_pair(points[j], -0.9));
				verified = false;
			} else {
				//std::cout << "the path contains " << ces.size() << " points not in the inductive invaraint\n";
				//if (ces.size() <= 10) {
					// Either update the invariant (critic) or update the controller (actor).
					//int ac;
					//if (fixable(game, points[j], ac, critic)) {
					//	train_examples.push_back(std::make_pair(points[j], ac));
					//} else {
					//	double from_score = getScore(critic, points[j]);
					//	if (from_score >= std::abs(score))
					//		counter_examples.push_back(std::make_pair(reach, from_score));
					//	else 
					//		counter_examples.push_back(std::make_pair(points[j], score));
					//}
					//verified = false;
					//counter_examples.push_back(std::make_pair(points[j], 0.9));
					//counter_examples.insert(counter_examples.end(), ces.begin(), ces.end());
					//if (ces.size() > 0)
					//	counter_examples.push_back(ces[0]);
				//} else {
				//	counter_examples.push_back(std::make_pair(points[j], -0.9));
				//}
				//verified = false;
			}
		}
	}
	return verified;
}

void sample_inv (Game& game, ComputationGraph& actor, 
					std::vector<std::pair<double,double>>& initset,
					unsafes& unsafeset,
					std::vector<std::pair<double, double>>& stateset,
					std::vector<std::pair<std::vector<double>, double>>& counter_examples) 
{
	// Sample something from intial set
	{
		int iteri = 0;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 generator (seed);
		std::uniform_real_distribution<double> uniform01(0.0, 1.0);
		std::vector<std::vector<double>> points;
		while (iteri < 500) {
			// Sample a point within the range of intersted.
			std::vector<double> point;
			for (int j = 0; j < initset.size(); j++) {
				double start = initset[j].first;
				double end = initset[j].second;

				double sv =  (end - start) * uniform01(generator) + start;
				point.push_back(sv);
			}
			//std::cout << "Samping a value in range (";
			//	for (int j = 0; j < point.size(); j++) {
			//	std::cout << point[j];
			//}
			//std::cout << ")\n";
			points.push_back(point);
			iteri++;
		}
		for (int j = 0; j < points.size(); j++) {
			counter_examples.push_back(std::make_pair(points[j], 0.9));
		}
	}
	// Sample something from unsafe set
	{
		int iteri = 0;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 generator (seed);
		std::uniform_real_distribution<double> uniform01(0.0, 1.0);
		std::vector<std::vector<double>> points;
		while (iteri < 500) {
			// Sample a point within the range of intersted.
			std::vector<double> point;
			// But firstly decide an unsafe region
			int i = rand() % unsafeset.size();
			for (int j = 0; j < game.inputs(); j++) {
				if (unsafeset[i].find(j) != unsafeset[i].end()) {
					double start = unsafeset[i][j].first;
					double end = unsafeset[i][j].second;

					double sv =  (end - start) * uniform01(generator) + start;
					point.push_back(sv);
				} else {
					double start = stateset[j].first;
					double end = stateset[j].second;

					double sv =  (end - start) * uniform01(generator) + start;
					point.push_back(sv);
				}
			}
			//std::cout << "Samping a value in range (";
			//	for (int j = 0; j < point.size(); j++) {
			//	std::cout << point[j];
			//}
			//std::cout << ")\n";
			points.push_back(point);
			iteri++;
		}
		for (int j = 0; j < points.size(); j++) {
			counter_examples.push_back(std::make_pair(points[j], -0.9));
		}
	}
	// Sample something from right in the middle
	{
		std::vector<std::vector<double>> ces;
		//for (int j = 0; j < 100; j++) {
		//	game.reset();
		//	game.perturbation();
		//	std::vector<double> istart = game.getGameState();
		//	bool res = agentEval (game, actor, 100, ces);
		//	if (res) { // Positive data collected.
		//		//for (int i = 0; i < ces.size(); i++) {
		//		//	counter_examples.push_back (std::make_pair(ces[i], 0.9));
		//		//}
		//		counter_examples.push_back (std::make_pair(istart, 0.9));
		//	} 
		//}
		//ces.clear();

		int iteri = 0;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 generator (seed);
		std::uniform_real_distribution<double> uniform01(0.0, 1.0);
		std::vector<std::vector<double>> points;
		while (iteri < 500) {
			// Sample a point within the range of intersted.
			std::vector<double> point;
			for (int j = 0; j < stateset.size(); j++) {
				double start = stateset[j].first;
				double end = stateset[j].second;

				double sv =  (end - start) * uniform01(generator) + start;
				point.push_back(sv);
			}
			points.push_back(point);
			iteri++;
		}
		for (int j = 0; j < points.size(); j++) {
			game.setGameState(points[j]);
			bool res = agentEval (game, actor, 1000, ces);
			if (!res) { // Negative data collected.
				//for (int i = 0; i < ces.size(); i++) {
				//	counter_examples.push_back (std::make_pair(ces[i], -0.9));
				//}
				counter_examples.push_back (std::make_pair(points[j], -0.9));
			} else 
				counter_examples.push_back (std::make_pair(points[j], 0.9));
		}
	}
}

void train_actor (ComputationGraph& graph, Network& network, std::vector<std::pair<std::vector<double>, int>> examples) {
	// Supervised learning to train the controller.
	auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.005, 0.001));
	RMSProp* rmsprop = prop.get();
	Solver solver( std::move(prop) );
	// First learn on the new keypoints
	int error;
	std::cout << "---------------------------------------------------------------\n";
	std::cout << "Training on " << examples.size() << " actor samples\n";
	int iteri = 0;
	do {
		error = 0;
		for (std::vector<std::pair<std::vector<double>,int>>::reverse_iterator i = examples.rbegin(); 
				i != examples.rend(); ++i) {
			const auto& result = graph.forward(data(i->first));
			int row, col;
			result.maxCoeff(&row,&col);
			error += (row == i->second? 0 : 1);
			if (row != i->second) {
				Vector errorCache = Vector::Zero( result.size() );
				for (int k = 0; k < result.size(); k++) {
					if (k == i->second) 
						errorCache[k] = result[k] - 1.0;
					else
						errorCache[k] = result[k] - (-1.0);
				}
				graph.backpropagate(errorCache, solver);
			}
		}
		network.update( solver );
		iteri ++;
		if (iteri % 1000 == 0)
			std::cout << "temporaral fix training error number: " << error << "\n";
	} while (error > 0 && iteri < 10000);
	std::cout << "fix training with error number: " << error << "\n";
}

int train_critic (ComputationGraph& graph, Network& network, std::vector<std::pair<std::vector<double>, double>> examples) {
	// Supervised learning to train the controller.
	auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.005, 0.001));
	RMSProp* rmsprop = prop.get();
	Solver solver( std::move(prop) );
	// First learn on the new keypoints
	int error;
	std::cout << "---------------------------------------------------------------\n";
	std::cout << "Training on " << examples.size() << " critic samples\n";
	int iteri = 0;
	do {
		error = 0;
		for (std::vector<std::pair<std::vector<double>,double>>::reverse_iterator i = examples.rbegin(); 
				i != examples.rend(); ++i) {
			const auto& result = graph.forward(data(i->first));
			if (i->second * result[0] <= 0) {
				error ++;
			}
			Vector errorCache = Vector::Zero( result.size() );
			errorCache[0] = result[0] - i->second;
			graph.backpropagate(errorCache, solver);
		}
		network.update( solver );
		iteri ++;
		if (iteri % 1000 == 0) 
			std::cout << "temporaral fix training error number: " << error << "\n";
	} while (error > 0 && iteri < 1000);
	std::cout << "fix training with error number: " << error << "\n";
	//if (error == examples.size()) {
	//	for (std::vector<std::pair<std::vector<double>,double>>::reverse_iterator i = examples.rbegin(); 
	//			i != examples.rend(); ++i) {
	//		for (int m = 0; m < i->first.size(); m++) {
	//			if (m != i->first.size() - 1)
	//				std::cout << i->first[m] << ",";
	//			else
	//				std::cout << i->first[m] << " : ";
	//		}
	//		std::cout << i->second << " / " << graph.forward(data(i->first))[0] << "\n";
	//	}
	//}
	return error;
}

// stateset is an esitimation of all rechable states. 
void correct_by_construction_train (Game& game, 
		ComputationGraph& actor_graph, Network& actor, 
		ComputationGraph& critic_graph, Network& critic,
		std::vector<std::pair<double,double>>& initset,
		unsafes& unsafeset,
		std::vector<std::pair<double, double>>& stateset) 
{
	std::vector<std::pair<std::vector<double>, double>> counter_examples;
	std::vector<std::pair<std::vector<double>, int>> train_examples;
	int n_samples = 10000;

	bool verified = true;
	int iteri = 0;

	do {
		sample_inv (game, actor_graph, initset, unsafeset, stateset, counter_examples);
		int error = 0;
		//do {
			error = train_critic (critic_graph, critic, counter_examples);
		//} while (error >= 3);
		iteri ++;

		std::vector<std::pair<std::vector<double>, double>> ces;
		bool prev = sample_to_check_precondition (game, actor_graph, critic_graph, n_samples, initset, ces, train_examples);
		std::cout << "counterexamples after precheck: " << ces.size() << "\n";
		bool postv = sample_to_check_postcondition (game, actor_graph, critic_graph, n_samples, unsafeset, stateset, ces, train_examples);
		std::cout << "counterexamples after postcheck: " << ces.size() << "\n";
		bool invv = sample_to_check_inv (game, actor_graph, critic_graph, n_samples, stateset, ces, train_examples);
		std::cout << "counterexamples after invcheck: " << ces.size() << "\n";
		verified = prev && postv && invv;
		counter_examples.clear();
	} while (iteri <= 10 && !verified); 

	//counter_examples.clear();

	//do {
	//	int o_s = counter_examples.size();
	//	//std::cout << "=========== Learing iteration: " << iteri << "\n";
	//	bool prev = sample_to_check_precondition (game, actor_graph, critic_graph, n_samples, initset, counter_examples, train_examples);
	//	std::cout << "counterexamples after precheck: " << counter_examples.size() - o_s << "\n";
	//	bool postv = sample_to_check_postcondition (game, actor_graph, critic_graph, n_samples, unsafeset, stateset, counter_examples, train_examples);
	//	std::cout << "counterexamples after postcheck: " << counter_examples.size() - o_s << "\n";
	//	bool invv = sample_to_check_inv (game, actor_graph, critic_graph, n_samples, stateset, counter_examples, train_examples);
	//	std::cout << "counterexamples after invcheck: " << counter_examples.size() - o_s << "\n";
	//	verified = prev && postv && invv;
	//	//train_actor (actor_graph, actor, train_examples);
	//	train_critic (critic_graph, critic, counter_examples);
	//	//train_examples.clear();
	//	//counter_examples.clear();
	//	iteri ++;
	//} while (!verified);
	std::cout << "Trained Successfully!\n";
}

void correct_by_construction_train (Game& game, 
		std::string agent,
		std::vector<std::pair<double,double>>& initset,
		unsafes& unsafeset,
		std::vector<std::pair<double, double>>& stateset) 
{
	//Network actor;
	//actor << FcLayer(Matrix::Random(10, game.inputs()).array() / 5);
	//actor << ReLULayer(Matrix::Zero(10, 1));
	//actor << FcLayer(Matrix::Random(10, 10).array() / 5);
	//actor << ReLULayer(Matrix::Zero(10, 1));
	//actor << FcLayer(Matrix::Random(game.actions(), 10).array() / 5);
	//actor << TanhLayer(Matrix::Zero(game.actions(), 1));

	Network actor;
	actor.load (agent);	
	ComputationGraph actor_graph(actor);

	Network critic;
	critic << FcLayer(Matrix::Random(10, game.inputs()).array() / 5);
	critic << ReLULayer(Matrix::Zero(10, 1));
	critic << FcLayer(Matrix::Random(10, 10).array() / 5);
	critic << ReLULayer(Matrix::Zero(10, 1));
	critic << FcLayer(Matrix::Random(1, 10).array() / 5);
	critic << TanhLayer(Matrix::Zero(1, 1));
	
	ComputationGraph critic_graph(critic);

	correct_by_construction_train(game, actor_graph, actor, critic_graph, critic, initset, unsafeset, stateset);

	//actor.save ("correct_by_construction_actor.network");
	critic.save ("correct_by_construction_critic.network");

	//agentPlay(game, "correct_by_construction_actor.network", 1000, 200);
}
