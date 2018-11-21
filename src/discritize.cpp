#ifndef GRID_H_
#define GRID_H_

//#include <Python.h>
#include <random>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <boost/functional/hash.hpp>

#include "config.h"
#include "Game.h"
#include "pole.h"
#include "drive.h"
#include "net/network.hpp"
#include "computation_graph.hpp"

using namespace net;

typedef std::unordered_map<int, std::pair<double,double>> partial_states; // conjunction of ranges
typedef std::vector<partial_states> unsafes;  // disjunction of conjunctions of ranges
typedef std::vector<std::pair<double,double>> whole_states;

template < typename SEQUENCE > struct seq_hash
{
    std::size_t operator() ( const SEQUENCE& seq ) const
    {
        std::size_t hash = 0 ;
        boost::hash_range( hash, seq.begin(), seq.end() ) ;
        return hash ;
    }
};

template < typename SEQUENCE, typename T >
using sequence_to_data_map = std::unordered_map< SEQUENCE, T, seq_hash<SEQUENCE> > ;

int binary_search(std::vector<double> v, double data) {
    auto it = std::upper_bound(v.begin(), v.end(), data);
    if (it == v.begin()) {
    	return 0;
    } else if (it == v.end()) {
    	return v.size()-2;
    } else {
    	std::size_t index = std::distance(v.begin(), it);
        return index-1;
    }   
}

class Grid {// abstract a world into grids
private:
	void enumerateStateConfiguration (int i, std::vector<std::vector<int>>& stateconfigs) {
		std::cout << "enumerateStateConfiguration i = " << i << "\n";
		if (i == dimensions.size() - 1) {
			for (int k = 0; k < dimensions[i].size() - 1; k++) {
				std::vector<int> nv;
				nv.push_back(k);
				stateconfigs.push_back(nv);
			}
			return;
		}

		enumerateStateConfiguration (i+1, stateconfigs);

		int n = stateconfigs.size();
		for (int j = 0; j < n; j++) {
			stateconfigs[j].insert (stateconfigs[j].begin(), 0);
		}
		for (int k = 1; k < dimensions[i].size() - 1; k++) {
			for (int j = 0; j < n; j++) {
				std::vector<int> nv;
				nv.insert(nv.begin(), stateconfigs[j].begin(), stateconfigs[j].end());
				nv[0] = k;
				stateconfigs.push_back(nv);
			}
		}
	}

	bool empty_range (std::pair<double,double> arg1, std::pair<double,double> arg2) {
		std::pair<double,double> intersection = { std::max(arg1.first, arg2.first), std::min(arg1.second, arg2.second) };
		return (intersection.second <= intersection.first);
	}

	bool check_range (whole_states st, whole_states pst) {
		for (int i = 0; i < st.size(); i++) {
			if (empty_range (st[i], pst[i])) 
				return false;
		}
		return true;
	}

	bool check_range (whole_states st, partial_states pst) {
		for (int i = 0; i < st.size(); i++) {
			if (pst.find(i) != pst.end() && empty_range (st[i], pst[i])) 
				return false;
		}
		return true;
	}

	bool check_range (whole_states st, unsafes usfs) {
		for (int i = 0; i < usfs.size(); i++) {
			if (check_range (st, usfs[i]))
				return true;
		}
		return false;
	}

	std::vector<double> stepGame (std::vector<double> gamestate) {
		game->setGameState (gamestate);
		
		/*Vector vec;
        vec.resize (game->inputs());
        for (int i = 0; i < game->inputs(); i++) {
        	vec[i] = gamestate[i];
        }
        const auto& result = graph->forward(vec);
        // greedy algorithm that generates the next action.
        int row, col;
        result.maxCoeff(&row,&col);
        game->step (row);*/

		double x = gamestate[0];
		double gamma = gamestate[1];
		int row;
  		if (x <= -1.21205854416)
    		row = 0;
  		else
    		if (gamma <= 0.137111499906)
      			row = 2;
    		else
      			if (x <= 0.387143492699)
        			row = 0;
      			else
        			row = 1;
        game->step(row);
        return game->getGameState();
	}

public:
	// e.g. x  : -0.1, 0, 0.1 yields (-0.1, 0), (0, 0.1) 
	// e.g. dx : -0.1, 0, 0.1 yields (-0.1, 0), (0, 0.1)
	// e.g. t  : -0.1, 0, 0.1 yields (-0.1, 0), (0, 0.1)
	std::vector<std::vector<double>> dimensions;

	std::unordered_map<int, std::vector<int>> index_to_coords;
	sequence_to_data_map<std::vector<int>, int> coords_to_index;

	std::vector<int> error_states;
	std::unordered_set<int> error_set;
	std::vector<int> init_states;

	// Number of states
	int S;
	// Number of actions
	int A;
	// Pointer to a game instance
	Game* game;
	// Pointer to a neurual network
	ComputationGraph* graph;

	Grid (Game& game, ComputationGraph& graph, 
			std::vector<std::vector<double>>& dimensions, whole_states& init_zones, unsafes& error_zones) {
		(this->dimensions).insert((this->dimensions).end(), dimensions.begin(), dimensions.end());
		this->game = &game;
		this->graph = &graph;
		this->A = game.actions();

		std::vector<std::vector<int>> stateconfigs;
		std::cout << "Discretizing state space ...\n";
		enumerateStateConfiguration (0, stateconfigs);
		std::cout << "State space discretized\n";
		S = 0;
		for (int i = 0; i < stateconfigs.size(); i++) {
			std::cout << "(";
			for (int ind : stateconfigs[i])
				std::cout << ind << " ";
			std::cout << ") is mapped to " << i << "\n";
			index_to_coords.insert (std::make_pair (S, stateconfigs[i]));
			coords_to_index.insert (std::make_pair (stateconfigs[i], S));
			S ++;

			whole_states state;
			for (int j = 0; j < stateconfigs[i].size(); j++) {
				double start = dimensions[j][stateconfigs[i][j]];
				double end = dimensions[j][stateconfigs[i][j]+1];
				state.push_back(std::make_pair(start,end));
			}
			if (check_range(state, error_zones)) {
				error_states.push_back(i);
				error_set.insert(i);
			}
			if (check_range(state, init_zones)) {
				init_states.push_back(i);
			}
		}
	}

	// An observation is converted to the state of our world.
	// e.g. (x:-002, dx:0.05, t : 0.1)
	int observation_to_index (std::vector<double> observation) {
		//std::cout << "Transitioning to (";
		//for (int j = 0; j < observation.size(); j++) {
		//	std::cout << observation[j];
		//}
		//std::cout << ")\n";

		std::vector<int> coord;
		for (int i = 0; i < observation.size(); i++) {
			double v = observation[i];
			int ind = binary_search (dimensions[i], v);
			coord.push_back(ind);
		}

		//std::cout << "Transitioning to in coords (";
		//for (int j = 0; j < coord.size(); j++) {
		//	std::cout << coord[j];
		//}
		//std::cout << ")\n";

		int index = coords_to_index[coord];
		return index;
	}

	// A state of our world is mapped back to a range of possible observations.
	std::vector<int> index_to_observation_range (int state) {
		return index_to_coords[state];
	} 

	// Build the tranision relations of states in our world.
	void build_transitions (int n_samples, std::unordered_map<int, std::unordered_map<int, double>>& transitionCounts) {
		std::cout << "Building probablisitic transitions about all states\n";
		for (int i = 0; i < S; i++) {
			if (error_set.find(i) != error_set.end()) // No tranisition exploration for unsafe states.
				continue;
			std::vector<int> ranges = index_to_observation_range(i);
			//std::cout << "Transitioning from (";
			//for (int j = 0; j < ranges.size(); j++) {
			//	std::cout << ranges[j];
			//}
			//std::cout << ")\n";
			int iteri = 0;
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    		std::mt19937 generator (seed);
    		std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    		std::vector<std::vector<double>> points;
			while (iteri < n_samples) {
				// Sample a point within the range of intersted.
				std::vector<double> point;
				for (int j = 0; j < ranges.size(); j++) {
					double start = dimensions[j][ranges[j]];
					double end = dimensions[j][ranges[j]+1];

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
			std::unordered_map<int, double> currTransitionCount;
			for (int j = 0; j < n_samples; j++) {
				int s = observation_to_index(stepGame(points[j]));
				//std::cout << "sampled a tranision from " << i << " to " << s << "\n";
				currTransitionCount[s]++;
			}
			for( auto& n : currTransitionCount ) {
        		n.second /= n_samples;
    		}
    		transitionCounts[i] = currTransitionCount;
		}
	}

	void outputTraisitionCounts (std::unordered_map<int, std::unordered_map<int, double>>& transitionCounts) {
		std::string datafile = "./data/optimal_policy";
		std::fstream evl(datafile, std::fstream::out);
		for (int i = 0; i < S; i++) {
			for (int j = 0; j < S; j++) {
				if (transitionCounts[i].find(j) != transitionCounts[i].end())
					evl << i << " " << j << " " << transitionCounts[i][j] << "\n";	
				else 
					evl << i << " " << j << " 0\n";	
			}
		}
		evl.close();
	}

	void outputInitial () {
		std::string datafile = "./data/start";
		std::fstream evl(datafile, std::fstream::out);
		for (int i = 0; i < init_states.size(); i++)
			evl << init_states[i] << "\n";
		evl.close();
	}

	void outputUnsafe () {
		std::string datafile = "./data/unsafe";
		std::fstream evl(datafile, std::fstream::out);
		for (int i = 0; i < error_states.size(); i++)
			evl << error_states[i] << "\n";
		evl.close();
	}

	void createSingleInitial (std::unordered_map<int, std::unordered_map<int, double>>& transitionCounts) {
		std::unordered_map<int, double> currTransitionCount;
		for (int i = 0; i < init_states.size(); i++) {
			currTransitionCount[init_states[i]] = ((double)1.0 / (init_states.size()));
		}
		transitionCounts[S] = currTransitionCount;
		S++;
	}	

	void createSingleFinal (std::unordered_map<int, std::unordered_map<int, double>>& transitionCounts) {
		for (int i = 0; i < error_states.size(); i++) {
			std::unordered_map<int, double> currTransitionCount;
			currTransitionCount[S] = 1.0;
			transitionCounts[error_states[i]] = currTransitionCount;
		}
		std::unordered_map<int, double> currTransitionCount;
		currTransitionCount[S] = 1.0;
		transitionCounts[S] = currTransitionCount;
		S++;
	}

	void outputStateSpace () {
		std::string datafile = "./data/state_space";
		std::fstream evl(datafile, std::fstream::out);
		evl << "states\n";
		evl << S << "\n";
		evl << "actions\n";
		evl << A;
		evl.close();
	} 

	// Verify the safety of our world.
	double verify (int n_samples, int steps) {
		std::cout << "Verify generated probablisitic system\n";
		outputInitial ();
		outputUnsafe ();

		std::unordered_map<int, std::unordered_map<int, double>> transitionCounts;
		build_transitions (n_samples, transitionCounts);
		// Create a single intial state and final state
		createSingleInitial(transitionCounts);
		createSingleFinal(transitionCounts);
		// export policy from transitionCounts
		outputTraisitionCounts (transitionCounts);

		// Output the state space.
		outputStateSpace();

		double verification_result = 0.0;
		
		// call prism to construct a probablisitic tranision system.
		/*setenv("PYTHONPATH", ".", 1);
		Py_Initialize();

		PyObject *pName = PyString_FromString("prism");
    	PyObject *pModule = PyImport_Import(pName);
    	Py_DECREF(pName);

    	if (pModule != NULL) {
        	PyObject *pFunc = PyObject_GetAttrString(pModule, "model_check");
        	if (pFunc && PyCallable_Check(pFunc)) {
        		PyObject *pArgs = PyTuple_New(2);
        		PyObject *pValue1 = PyInt_FromLong(steps);
        		PyObject *pValue2 = PyInt_FromLong(S);
        		PyTuple_SetItem(pArgs, 0, pValue1);
        		PyTuple_SetItem(pArgs, 1, pValue2);

            	PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
            	Py_DECREF(pArgs);
            	if (pValue != NULL) {
            		verification_result = PyFloat_AsDouble(pValue);
                	printf("Result of call: %lf\n", verification_result);
                	Py_DECREF(pValue);
            	}
            	else {
                	Py_DECREF(pFunc);
                	Py_DECREF(pModule);
                	PyErr_Print();
                	fprintf(stderr,"Call Prism model_check failed\n");
                	return 0;
            	}
        	} else {
            	if (PyErr_Occurred())
                	PyErr_Print();
            	fprintf(stderr, "Cannot find model_check function\n");
            	return 0;
        	}
        	Py_XDECREF(pFunc);
        	Py_DECREF(pModule);
        } else {
        	PyErr_Print();
        	fprintf(stderr, "Failed to load prism\n");
        	return 0;
        }
        Py_Finalize();*/

        return verification_result;
	}

};

int main_verify(int argc, char** argv)
{
	std::string model(argv[1]);
	Network agent;
	agent.load (model);
	ComputationGraph graph(agent);
	//Pole pole; 
	Drive pole;
	std::vector<std::vector<double>> dimensions;
	//double xd[] = {-1.2, -1.0, 0, 1.0, 1.2};
	//double xd[] = {-1.2,-1.0,-0.5,-0.4,-0.2,-0.000001,0.000001,0.2,0.4,0.5,1.0,1.2};
	// -- _that_is_within_safe_angle_ -- double xd[] = {-1.2,-1.0,-0.5,-0.4,-0.3,-0.2,-0.1,-0.000001,0.000001,1.2};
	// -- nice-abstraction - double xd[] = {-1.2,-1.0,-0.9,-0.7,-0.5,-0.3,-0.1,-0.000001,0.000001,1.2};
	//double xd[] = {-1.2,-1.0,-0.9,-0.7,-0.5,-0.3,-0.1,-0.000001,0.000001,1.2};
	double xd[] = {-2.2,-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.21205854416,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.008,-0.006,-0.004,-0.002,0,0.002,0.004,0.006,0.008,0.1,0.2,0.387143492699,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2};
  	std::vector<double> dx (xd, xd + sizeof(xd) / sizeof(double));
  	//double dxd[] = {-1.0, -0.2, 0, 0.2, 1.0};
  	//double dxd[] = {-1.0,-0.09,-0.003,-0.002,-0.000001,0.000001,0.002,0.003,0.09,1.0};
  	// -- _that_is_within_safe_angle_ -- double dxd[] = {-1.0,-0.09,-0.005,-0.003,-0.002,-0.000001,0.000001,1.0};
  	// -- nice-abstraction - double dxd[] = {-0.1,-0.09,-0.009,-0.005,-0.003,-0.002,-0.000001,0.000001,0.1};
  	//double dxd[] = {-0.2,-0.1,-0.009,-0.005,-0.003,-0.002,-0.000001,0.000001,0.2};
  	double dxd[] = {-1.3,-1.2,-1.1,-1,-0.9,-0.8,-0.78539815,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.008,-0.006,-0.004,-0.002,0,0.002,0.004,0.006,0.008,0.137111499906,0.2,0.3,0.4,0.5,0.6,0.78539815,0.8,0.9,1,1.1,1.2,1.3};
  	std::vector<double> ddx (dxd, dxd + sizeof(dxd) / sizeof(double));
  	//double thetad[] = {-1.0, -0.026179938765, 0, 0.026179938765, 1.0};
  	//double thetad[] = {-1.0,-0.026179938765,-0.005,-0.003,-0.002,-0.000001,0.000001,0.002,0.003,0.005,0.026179938765,1.0};
  	// -- _that_is_within_safe_angle_ -- double thetad[] = {-1.0,-0.026179938765,-0.01,-0.003,-0.002,-0.000001,0.000001,1.0};
  	// -- nice-abstraction - double thetad[] = {-0.1,-0.026179938765,-0.01,-0.003,-0.002,-0.000001,0.000001,0.1};
  	//*** double thetad[] = {-0.1,-0.026179938765,-0.01,-0.003,-0.002,-0.000001,0.000001,0.1};
  	//*** std::vector<double> dtheta (thetad, thetad + sizeof(thetad) / sizeof(double));
  	//double dthetad[] = {-1.0, -0.02, 0, 0.02, 1.0};
  	//double dthetad[] = {-1.0,-0.1,-0.05,-0.02,-0.01745329251,0.01745329251,0.02,0.05,0.1,1.0};
  	// -- _that_is_within_safe_angle_ -- double dthetad[] = {-1.0,-0.1,-0.05,-0.02,-0.01745329251,0.01745329251,1.0};
  	// -- nice-abstraction - double dthetad[] = {-0.1,-0.05,-0.02,-0.01745329251,0.01745329251,0.1};
  	// *** double dthetad[] = {-0.1,-0.05,-0.02,-0.01745329251,0.01745329251,0.1};
  	// *** std::vector<double> ddtheta (dthetad, dthetad + sizeof(dthetad) / sizeof(double));
	dimensions.push_back(dx);
	dimensions.push_back(ddx);
	//dimensions.push_back(dtheta);
	//dimensions.push_back(ddtheta);

	unsafes error_zones;
	//partial_states e1;
	//e1[0] = std::make_pair(-2.2, -2.0);
	//error_zones.push_back(e1);
	partial_states e2;
	e2[0] = std::make_pair(2.0, 2.2);
	error_zones.push_back(e2);	
	//partial_states e1;
	//e1[2] = std::make_pair(-1.0, -0.026179938765);
	//error_zones.push_back(e1);
	//partial_states e2;
	//e2[2] = std::make_pair(0.026179938765, 1.0);
	//error_zones.push_back(e2);
	//partial_states e3;
	//e3[0] = std::make_pair(-1.2, -1.0);
	//error_zones.push_back(e3);
	//partial_states e4;
	//e4[0] = std::make_pair(1.0, 1.2);
	//error_zones.push_back(e4);

	whole_states init_zones;
	init_zones.push_back (std::make_pair(-1,1));
	init_zones.push_back (std::make_pair(-0.78539815,0.78539815));
	//init_zones.push_back (std::make_pair(-0.000001,0.000001));
	//init_zones.push_back (std::make_pair(-0.000001,0.000001));
	//init_zones.push_back (std::make_pair(-0.000001,0.000001));
	//init_zones.push_back (std::make_pair(-0.01745329251,0.01745329251));

	Grid g(pole, graph, dimensions, init_zones, error_zones);
	g.verify (1000, 200);
	return 0;
}

#endif /* GRID_H_ */