// Main verification engine of neural networks
//===================  Test and Verification of learned neural model =================== 
#include "qlearner/action.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <climits>

#include "config.h"
#include "net/fc_layer.hpp"
#include "net/relu_layer.hpp"
#include "net/tanh_layer.hpp"
#include "net/solver.hpp"
#include "net/rmsprop.hpp"
#include "net/network.hpp"

#include "dt.h"
#include "Game.h"

#include "pole.h"
#include "pong.h"

#include <random>

using namespace net;
using namespace qlearn;

typedef std::vector<double> gamestate;

Vector vec;

// Encode game state using vector
const Vector& data(gamestate cpst)
{
	vec.resize (cpst.size());
	for (int i = 0; i < cpst.size(); i++) {
		vec[i] = cpst[i];
	}
	return vec;
}

void display(gamestate cpst)
{	
	for (int i = 0; i < cpst.size(); i++) {
		if (i != cpst.size() - 1)
			std::cout << cpst[i] << ",";
		else
			std::cout << cpst[i] << "\n";
	}
}

void display(gamestate cpst, int ac)
{	
	for (int i = 0; i < cpst.size(); i++) {
		if (i != cpst.size() - 1)
			std::cout << cpst[i] << ",";
		else
			std::cout << cpst[i] << " : ";
	}
	std::cout << ac << "\n";
}

void render(std::fstream& out, gamestate cpst, int ac)
{
	for (double d : cpst) {
		out << d << ",";
	}
	out << "Class" << ac << "\n";
}

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
void agentPlay (Game& game, std::string filename, int times) {
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
		while(true)
		{
			auto ac = getAction(graph, data(game.getGameState()));
			//std::cout << s << ": "; display (game.getGameState(), ac.id);
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
		if (terminated) succ++;
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
		gamestate st = game.getGameState();
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
			int ac = uct_advise(sim, i->first, game.fail()||game.terminate());
			if (ac == i->second) {
				// The mc and neural net agrees with each other.
			} else {
				std::vector<std::pair<gamestate,int>> local_keyset;
				gamestate st = i->first;
				game.setGameState (st);
				int i = 1;
				bool res = false;
				while (i <= bound - s) {
					local_keyset.push_back (std::make_pair (st, ac));
					game.step(ac);
					if (game.fail()) break;
					if (game.terminate()) { res = true; break; }
					st = game.getGameState();
					//std::cout << "agent needs to play " << bound - s - i << " steps from "; display (st);
					res = agentPlay (game, graph, bound - s - i);
					game.setGameState(st);
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

	std::vector<std::pair<gamestate,int>> keyset;

	int g = 0;
	do {
		int size = keyset.size();
		bool fixable = key_gen (game, sim, graph, bound, keyset);
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

	agentPlay(game, "supervised_agent.network", 100);
}

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
		path.push_back (std::make_pair(st, ac));
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
			game.setGameState (i->first);
			// Check how neural net works.
			auto ac = getAction(graph, data(i->first));
			if (ac.id == i->second) {
				// The abstraction and neural net agrees with each other.
			} else {
				// Check if neural net's decision is good.
				bool res = agentPlay (game, graph, bound - s);
				game.setGameState(i->first);
				if (res) {
					keyset.push_back (std::make_pair (i->first, ac.id));
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
void abstraction_check (Game& game, std::string filename, int bound) {
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
				if (keyset.size() > 200) {
					std::cout << "keyset.size() == " << keyset.size() << "\n";
					dt.learn(datafile);
				}
			}
		}
	} while (g < 10); // Verification converges after consecutive success in several rounds.
	std::cout << "Verifed!\n";

	int ts = 0;
	int succ = 0;
	for(int g = 0; g < 100; ++g)
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
			ts ++;
			if( failed || terminated) {
				std::cout << "Maintained in " << s << " steps\n";
				break;
			}
		}
		if (s >= bound || terminated) 
			succ++;
	}
	std::cout << (ts / 100.0) << "\n";
	std::cout << succ << " out of 100 games is successful.\n";
} 

void testNetwork () {
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

int main(int argc, char** argv)
{
	if (argc > 1) {
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
			if (gamename.compare ("pong") == 0) {
				Pong game;
				UCTPongSimulator sim;
				cegis_train(game, sim, INT_MAX);
			} else if (gamename.compare ("pole") == 0) {
				Pole game;
				UCTPoleSimulator sim;
				cegis_train(game, sim, 200);
			} else {
				std::cout << "The game " << gamename << "is not found.\n";
				return -1;
			}
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
			//std::cout << "decision : " << ac.id << "\n";

			//agentPlay (param);
			//networkToJson(agentfile);
			
			// -- doing abstraction refinement --
			Pole game;
			abstraction_check (game, param, 200);
		}
		return 0;
	}
}