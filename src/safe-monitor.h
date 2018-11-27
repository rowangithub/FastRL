#ifndef SAFE_MONITOR_H_
#define SAFE_MONITOR_H_

/** Construct a DT approximation of safe set and policy */

#include "qlearner/action.h"

#include "config.h"
#include "dt.h"
#include "Game.h"
#include "verify.h"

using namespace net;
using namespace qlearn;

typedef std::vector<double> gamestate;

void prepare_keyset (Game& game, ComputationGraph& actor, ComputationGraph& critic, 
		std::vector<std::pair<gamestate,int>>& keyset_controller,
		std::vector<std::pair<gamestate,int>>& keyset_safeset,
		std::vector<std::pair<double, double>> stateset,
		int n_bootstrap) 
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator (seed);
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);

	int c = 0;
	int c2 = 0;
	int* kind = new int[game.actions()];
	std::vector<gamestate> kindv;
	for (int i = 0; i < game.actions(); i++) kind[i] = -1;
	int* kind2 = new int[2];
	std::vector<gamestate> kind2v;
	kind2[0] = -1; kind2[1] = -1;
	

	while (1) {
		// Sample a point within the range of interested.
		std::vector<double> point;
		for (int j = 0; j < stateset.size(); j++) {
			double start = stateset[j].first;
			double end = stateset[j].second;

			double sv =  (end - start) * uniform01(generator) + start;
			point.push_back(sv);
		}

		auto ac = getAction(actor, data(point));
		const auto& val = critic.forward(data(point));

		if (kind[ac.id] == -1) {
			kind[ac.id] = c;
			//keyset_controller.push_back(std::make_pair (point, ac.id));
			kindv.push_back(point);
			c++;
		}
		if (val[0] <= 0 && kind2[0] == -1) {
			kind2[0] = c2;
			//keyset_safeset.push_back(std::make_pair (point, 0));
			kind2v.push_back(point);
			c2++;
		}
		if (val[0] > 0 && kind2[1] == -1) {
			kind2[1] = c2;
			//keyset_safeset.push_back(std::make_pair (point, 1));
			kind2v.push_back(point);
			c2++;
		}
		if (c == game.actions() && c2 == 2) {
			for (int i = 0; i < game.actions(); i++)
				keyset_controller.push_back(std::make_pair(kindv[kind[i]], i));
			for (int i = 0; i < 2; i++)
				keyset_safeset.push_back(std::make_pair(kind2v[kind2[i]], i));
			break;
		}
	}

	delete [] kind;
	delete [] kind2;

	//Sample more points in safe-set to bootstrap the whole process 
	{
		int iteri = 0;
		while (iteri < n_bootstrap - 2) {
			std::vector<double> point;
			for (int j = 0; j < stateset.size(); j++) {
				double start = stateset[j].first;
				double end = stateset[j].second;

				double sv =  (end - start) * uniform01(generator) + start;
				point.push_back(sv);
			}

			const auto& val = critic.forward(data(point));
			keyset_safeset.push_back(std::make_pair(point, val[0] > 0 ? 1 : 0));
			iteri++;
		}
	}
}

bool agentTrail (Game& game, ComputationGraph& actor, ComputationGraph& critic, int bound, int& ces) 
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
			const auto& result = critic.forward(data(game.getGameState()));
			double score = result[0];
			if (score <= 0) ces++;
		}
	}
	//std::cout << "agent played " << s << " step\n";
	return (s == bound || terminated);
}

bool train_monitor (Game& game, ComputationGraph& actor,      // Given a game and policy described in game and graph
		ComputationGraph& critic,						      // together with an estimation of safe set
		DT& dt_controller, DT& dt_safeset, 			   // DT approximations for both controller and safeset
		std::vector<std::pair<gamestate,int>>& keyset_controller, // training set for the controller approximation
		std::vector<std::pair<gamestate,int>>& keyset_safeset, 	  // training set for the safeset approximation
		int n_samples, std::vector<std::pair<double, double>> stateset, // training is by sampling from an user-given domain
		int iteration
) 
{
	// Sample from dt_safeset and check whether it is already inductive
	// For a counterexample to induction, either increase keyset_controller or keyset_safeset
	// Retrain the DT approximations if necessary.

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
		
		double*  dtst = new double[point.size()];
		for (int i = 0; i < point.size(); i++) {
			dtst[i] = point[i];
		}
		int res = dt_safeset.predict (dtst, point.size());
		delete [] dtst;

		if (res > 0)
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
			keyset_safeset.push_back(std::make_pair(points[j], 0));
			//std::cout << "counterexample to safeset because it includs a game-fail state\n";
			//display(points[j]);
			verified = false;
		} else {
			int actor_ac = (getAction(actor, data(points[j]))).id;
			double*  dtst = new double[points[j].size()];
			for (int i = 0; i < points[j].size(); i++) {
				dtst[i] = points[j][i];
			}
			int dt_ac = dt_controller.predict (dtst, points[j].size());
			delete [] dtst;

			game.step(dt_ac);
			std::vector<double> reach = game.getGameState();
			dtst = new double[reach.size()];
			for (int i = 0; i < reach.size(); i++) {
				dtst[i] = reach[i];
			}
			int res = dt_safeset.predict (dtst, reach.size());
			delete [] dtst;

			if (res <= 0) {
				verified = false;
				// Either update the dt-safeset or update the dt-controller.
				// First check if following actor can solve the problem.
				//  if solved, unpdate dt-safeset
				//  if not, go to the sceond step.
				// Second check if the state obtained by following actor is a nice state.
				// 	if it is not a nice state, remove points[j] from safeset.
				//  if it is a nice state, add 'reach' into safeset and update dt-safeset (if needed).
					
				// First Step:			
				if (actor_ac != dt_ac) {
					game.setGameState(points[j]);
					game.step(actor_ac);
					reach = game.getGameState();
					dtst = new double[reach.size()];
					for (int i = 0; i < reach.size(); i++) {
						dtst[i] = reach[i];
					}
					res = dt_safeset.predict (dtst, reach.size());
					delete [] dtst;
					if (res) {
						keyset_controller.push_back (std::make_pair(points[j], actor_ac));
						continue;
					}
				}
				// Second Step:
				//game.setGameState(reach);
				//int ces;
				//bool exe_res = agentTrail (game, actor, critic, 1000, ces);
				const auto& result = critic.forward(data(reach));
				if (result[0] <= 0.0) { // reach is not nice!
					keyset_safeset.push_back(std::make_pair(points[j], 0));
					//std::cout << "counterexample to safeset because it contains a not-nice state\n";
					//display(reach);
				} else {       			    // reach is nice!
					keyset_safeset.push_back(std::make_pair(reach, 1));
					//std::cout << "counterexample to safeset because it should contain a nice state\n";
					//display(reach);
					if (actor_ac != dt_ac) 
						keyset_controller.push_back (std::make_pair(points[j], actor_ac));
				}
			}
		}
	}
	return verified;
}

void train_monitor (Game& game, std::string actor_filename, std::string critic_filename, std::vector<std::pair<double, double>> stateset, 
		int n_samples, 	  // number of samples used to check DT approaximation correctness at a time
		int n_bootstrap,  // number of samples used to bootstrap the DT approximation of a set state set
		int n_stops
) 
{
	Network actor;
	actor.load (actor_filename);
	std::string actor_modelname = actor_filename + ".json";
	actor.saveJson (actor_modelname);
	ComputationGraph actor_graph(actor);
	DT dt_controller (actor_modelname);
	std::string datafile_controller = actor_filename+".data";
	std::remove(datafile_controller.c_str());

	Network critic;
	critic.load (critic_filename);
	std::string critic_modelname = critic_filename + ".json";
	critic.saveJson (critic_modelname);
	ComputationGraph critic_graph(critic);
	DT dt_safeset (critic_modelname);
	std::string datafile_safeset = critic_filename+".data";
	std::remove(datafile_safeset.c_str());

	std::vector<std::pair<gamestate,int>> keyset_controller; // training set for the controller approximation
	std::vector<std::pair<gamestate,int>> keyset_safeset; 	  // training set for the safeset approximation

	int sample_size_controller = 0;
	int sample_size_safeset = 0;

	prepare_keyset (game, actor_graph, critic_graph, keyset_controller, keyset_safeset, stateset, n_bootstrap);

	bool verified = true;
	int k = 0;
	int g = 0;
	do {
		std::cout << "Collected " << (keyset_controller.size() - sample_size_controller) << " examples for building the DT controller\n";
		std::cout << "Collected " << (keyset_safeset.size() - sample_size_safeset) << " examples for building the DT safeset\n";
		if (keyset_controller.size() != sample_size_controller) {
			std::fstream evl(datafile_controller, std::fstream::out);
			for (std::vector<std::pair<gamestate,int>>::iterator i = keyset_controller.begin(); 
				i != keyset_controller.end(); ++i) { 
				render (evl, i->first, i->second);
				evl.flush ();
			}
			evl.close();
			dt_controller.learn(datafile_controller, game.inputs(), 31);
		}
		if (keyset_safeset.size() != sample_size_safeset) {
			std::fstream evl(datafile_safeset, std::fstream::out);
			for (std::vector<std::pair<gamestate,int>>::iterator i = keyset_safeset.begin(); 
				i != keyset_safeset.end(); ++i) { 
				render (evl, i->first, i->second);
				evl.flush ();
			}
			evl.close();
			dt_safeset.learn(datafile_safeset, game.inputs(), 31);
		}
		sample_size_controller = keyset_controller.size();
		sample_size_safeset = keyset_safeset.size();
		verified = train_monitor (game, actor_graph, critic_graph, dt_controller, dt_safeset, keyset_controller, keyset_safeset, n_samples, stateset, k);
		if (verified) g++;
		else g = 0;
		std::cout << "Completed the " << k << " iteration of building a safe-monitor with " << g << " consecutive suceesses.\n";
		k++;
	} while (g < n_stops); //(!verified);
	std::cout << "Verifed!\n";
	// Abstraction to file for hybrid system verification!
	dt_controller.interprete(actor_filename+".abstraction", game.inputs());
	std::cout << "Testing learned abstraction:\n";
	int ts = 0;
	int succ = 0;
	int bound = 1000;
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
			int ac = dt_controller.predict (dtst, game.inputs());
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

#endif /* SAFE_MONITOR_H_ */