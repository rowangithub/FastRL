#ifndef __UCT_AGENT_H_
#define __UCT_AGENT_H_

#include "config.h"
#include "computation_graph.hpp"
#include "uct.hpp"

#include "pole.h"
#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"
#include "boost/tuple/tuple_io.hpp"


class UCTPoleState : public UCT::State {
public:    
    boost::tuples::tuple<double,double,double,double> polestate;

    UCTPoleState (boost::tuples::tuple<double,double,double,double> polestate) {
        setState (polestate);
    }

    virtual bool equal(UCT::State* state) {
        const UCTPoleState* other = dynamic_cast<const UCTPoleState*> (state);
        if ((other == NULL) || 
            (other->polestate.get<0>() != polestate.get<0>()) || 
            (other->polestate.get<1>() != polestate.get<1>()) || 
            (other->polestate.get<2>() != polestate.get<2>()) ||
            (other->polestate.get<3>() != polestate.get<3>())) {
            return false;
        }
        return true;
    }

    virtual UCT::State* duplicate() {
        UCTPoleState * other = new UCTPoleState(polestate);
        return other;
    }

    virtual void print() const {
        std::cout << "("
                  << polestate.get<0>() << " " 
                  << polestate.get<1>() << " " 
                  << polestate.get<2>() << " " 
                  << polestate.get<3>() << ")";
    }

    void setState (boost::tuples::tuple<double,double,double,double> st) {
        get<0>(polestate) = st.get<0>();
        get<1>(polestate) = st.get<1>();
        get<2>(polestate) = st.get<2>();
        get<3>(polestate) = st.get<3>();
    }	
};

class UCTPoleAction: public UCT::SimAction {
public:
    int id;
    UCTPoleAction(int _id): id(_id) {
    }

    virtual SimAction* duplicate() {
        UCTPoleAction* other = new UCTPoleAction(id);

        return other;
    }

    virtual void print() const {
        cout << id ;
    }

    virtual bool equal(SimAction* other) {
        UCTPoleAction* act = dynamic_cast<UCTPoleAction*>(other);
        return act->id == id;
    }
};

class UCTPoleSimulator : public UCT::Simulator {
public:
    // Game engine
    Pole pole_;
    // Game AI
    net::ComputationGraph* policy;

    UCTPoleState* current = NULL;
    vector<UCT::SimAction*> actVect;

    UCTPoleSimulator () {
        // Manage the interaction between agent and env
        pole_.reset ();
        pole_.perturbation ();

        // Construct actVect and current
        actVect.push_back (new UCTPoleAction (-1));
        actVect.push_back (new UCTPoleAction (0));
        actVect.push_back (new UCTPoleAction (1));
        current = new UCTPoleState (pole_.getState());

        this->policy = NULL;
    }

    UCTPoleSimulator (net::ComputationGraph* policy) {
        // Manage the interaction between agent and env
        pole_.reset ();
        pole_.perturbation ();

        // Construct actVect and current
        actVect.push_back (new UCTPoleAction (-1));
        actVect.push_back (new UCTPoleAction (0));
        actVect.push_back (new UCTPoleAction (1));
        current = new UCTPoleState (pole_.getState());

        this->policy = policy;
    }

    ~UCTPoleSimulator () {
        // free actVect and current
	delete actVect[0];
	delete actVect[1];
	delete actVect[2];
	delete current; 
    }

    virtual void setState (UCT::State* state) {
        const UCTPoleState* other = dynamic_cast<const UCTPoleState*> (state);
        if (other == NULL) {
            return;
        }
        current->setState (other->polestate);
        pole_.setState (other->polestate);
    }

    virtual UCT::State* getState () {
        return current;
    }

    // Return the index of the action that should be used for mc sampling.
    virtual int MC_action () {
        if (policy == NULL) {
            return rand() % actVect.size();
        } else {
            Vector vec;
            vec.resize (4);
            vec[0] = (current->polestate).get<0>();
            vec[1] = (current->polestate).get<1>();
            vec[2] = (current->polestate).get<2>();
            vec[3] = (current->polestate).get<3>();
            const auto& result = policy->forward(vec);
            // greedy algorithm that generates the next action.
            int row, col;
            result.maxCoeff(&row,&col);
            return row;
        }
    }

    virtual double act(const UCT::SimAction* action) {
        //assert(!isTerminal());

        const UCTPoleAction* act = dynamic_cast<const UCTPoleAction*> (action);
        if (act == NULL) {
            return 0;
        }
        int id = act->id;
        //if (rand() / (double) RAND_MAX < 0.1) {
        //    id = rand() % 4;
        //}
        pole_.setState (current->polestate);
        pole_.step (id);
        current->setState (pole_.getState());
        return pole_.fail() ? -1 : 1;
    }

    virtual vector<UCT::SimAction*>& getActions () {
        return actVect;
    }

    virtual bool isTerminal () {
        return (pole_.fail());
    }

    virtual void reset () {
        pole_.reset ();
        pole_.perturbation ();
        current->setState (pole_.getState());
    }
}; 

/**
int main (int argc, char** argue) {
    UCTPoleSimulator* sim = new UCTPoleSimulator ();
    UCTPoleSimulator* sim2 = new UCTPoleSimulator ();
    UCT::UCTPlanner uct (sim2, -1, 1000, 1, 0.95);
    int numGames = 1;

    for (int i = 0; i < numGames; ++i) {
        int steps = 0;
        double r = 0;
        sim->getState()->print();
        while (! sim->isTerminal()) {
            steps++; //std::cout << "step : " << steps << "\n";
            uct.setRootNode(sim->getState(), sim->getActions(), r, sim->isTerminal());
            uct.plan();
            UCT::SimAction* action = uct.getAction();

             cout << "-" ;
             action->print();
             cout << "->";
             uct.testTreeStructure();

            r = sim->act(action);
             sim->getState()->print();
             cout << endl;
             sim->getState()->print();
        }
        sim->reset();
        cout << "Game:" << i << "  steps: " << steps << "  r: " << r << endl;
    }
} */


/*class UCTAgent : public EpsilonAgent {

public:
	MonteCarloAgent(const bool test): EpsilonAgent(test) {
		monte_carlo_.load("monte-carlo.txt");
	}

	virtual ~MonteCarloAgent() {
		if (!test()) {
			monte_carlo_.save("monte-carlo.txt");

		}
	}

	double & qvalue(const State &, const int &);

	virtual void learn(const State & pre_state, int pre_action, double reward, const State &, int);
	virtual void fail(const State & state, int action, double reward);


}
*/

#endif