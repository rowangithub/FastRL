#ifndef GAME_H_
#define GAME_H_

#include <vector>

#include "config.h"
#include "uct.hpp"
#include "computation_graph.hpp"
#include "dt.h"

class UCTGameSimulator;

// An uniform inteface defineing game APIs.
class Game {
public:
	virtual bool terminate() const = 0;
	virtual bool fail() const = 0;
	virtual double measureFailure () const = 0;

	virtual void perturbation() = 0;
    
	virtual void reset()  = 0;

	virtual void step(int action) = 0;

	virtual int actions() = 0;
	virtual int inputs() = 0;

    // Get and Set for simulator to access internal game state
	virtual std::vector<double> getGameState () = 0;
	
    virtual void setGameState (std::vector<double> st) = 0;

    virtual void printGameState() = 0;

    /* -------------- Begin -- used for cegis-inv --------------*/
    // Get and Set for uct to access internal game state
    // This is redundant to getGameState usually but a place to let uct access more information in the context of cegis-inv.
    std::vector<double> getUCTState () {
        //std::vector<double> data;
        //data.push_back(x);
        //data.push_back(gamma);
        std::vector<double> data = getGameState();
        if (inv_mode)
            data.push_back(game_step);
        return data;          
    }

    void setUCTState (std::vector<double> st) {
        //x = st[0];
        //gamma = st[1];
        setGameState(st);
        if (inv_mode)
            game_step = st[inputs()];
    }

    bool inv_mode = false;
    int game_step = 0;
    /* -------------- End -- used for cegis-inv --------------*/
};

class UCTGameState : public UCT::State {
public:    
    std::vector<double> gstate;

    UCTGameState (std::vector<double> gstate) {
        setState (gstate);
    }

    virtual bool equal(UCT::State* state) {
        if (state == NULL)
        	return false;

        const UCTGameState* other = dynamic_cast<const UCTGameState*> (state);
        bool result = true;
        for (int i = 0; i < gstate.size(); i++) {
        	if (i >= other->gstate.size() || other->gstate[i] != gstate[i]) {
        		result = false;
        		break;
        	}
        }
        return result;
    }

    virtual UCT::State* duplicate() {
        UCTGameState * other = new UCTGameState(gstate);
        return other;
    }

    virtual void print() const {
    	std::cout << "(";
    	for (int i = 0; i < gstate.size(); i++) {
    		if (i != gstate.size() - 1)
    			std::cout << gstate[i] << " ";
    		else 
    			std::cout << gstate[i] << ")";
    	}
    }

    void setState (std::vector<double> st) {
    	gstate.clear();
    	gstate.insert(gstate.end(), st.begin(), st.end());
    }	
};

class UCTGameAction: public UCT::SimAction {
public:
    int id;
    UCTGameAction(int _id): id(_id) {
    }

    virtual SimAction* duplicate() {
        UCTGameAction* other = new UCTGameAction(id);

        return other;
    }

    virtual void print() const {
        cout << id ;
    }

    virtual bool equal(SimAction* other) {
        UCTGameAction* act = dynamic_cast<UCTGameAction*>(other);
        return act->id == id;
    }
};

class UCTGameSimulator : public UCT::Simulator {
public:
    // Game engine
    Game* game = NULL;
    // Game AI
    net::ComputationGraph* policy = NULL;
    DT* dt = NULL;

    UCTGameState* current = NULL;
    vector<UCT::SimAction*> actVect;

    void init (Game* game) {
    	assert (this->game == NULL);
    	this->game = game;
        // Manage the interaction between agent and env
        game->reset ();
        game->perturbation ();

        // Construct actVect and current
        for (int i = 0; i < game->actions(); i++) {
        	actVect.push_back (new UCTGameAction (i));
        }

        current = new UCTGameState (game->getUCTState());

        this->policy = NULL;
        this->dt = NULL;
    }

    void init (Game* game, net::ComputationGraph* policy) {
    	assert (this->game == NULL);
    	this->game = game;
        // Manage the interaction between agent and env
        game->reset ();
        game->perturbation ();

        // Construct actVect and current
        for (int i = 0; i < game->actions(); i++) {
        	actVect.push_back (new UCTGameAction (i));
        }
        current = new UCTGameState (game->getUCTState());

        this->policy = policy;
    }

    virtual ~UCTGameSimulator () {
    	if (game == NULL)
    		return;

        // free actVect and current
        for (int i = 0; i < game->actions(); i++) {
        	delete actVect[i];
        }
		delete current; 
    }

    virtual void setState (UCT::State* state) {
    	assert (game != NULL);
        const UCTGameState* other = dynamic_cast<const UCTGameState*> (state);
        if (other == NULL) {
            return;
        }
        current->setState (other->gstate);
        game->setUCTState (other->gstate);
    }

    virtual UCT::State* getState () {
        return current;
    }

    // Return the index of the action that should be used for mc sampling.
    virtual int MC_action () {
    	assert (game != NULL);
        if (policy == NULL && dt == NULL) {
            return rand() % actVect.size();
        } else if (policy != NULL) {
            Vector vec;
            vec.resize (game->inputs());
            for (int i = 0; i < game->inputs(); i++) {
            	vec[i] = (current->gstate)[i];
            }
            const auto& result = policy->forward(vec);
            // greedy algorithm that generates the next action.
            int row, col;
            result.maxCoeff(&row,&col);
            return row;
        } else { // DT is not NULL
            double*  dtst = new double[game->inputs()];
            for (int i = 0; i < game->inputs(); i++) {
                dtst[i] = (current->gstate)[i];
            }
            int ac = dt->predict (dtst, game->inputs());
            delete [] dtst;
            return ac;
        }
    }

    virtual double act(const UCT::SimAction* action) {
        //assert(!isTerminal());
        assert (game != NULL);

        const UCTGameAction* act = dynamic_cast<const UCTGameAction*> (action);
        if (act == NULL) {
            return 0;
        }
        int id = act->id;
        //if (rand() / (double) RAND_MAX < 0.1) {
        //    id = rand() % 4;
        //}
        game->setUCTState (current->gstate);
        game->step (id);
        current->setState (game->getUCTState());
        return game->fail() ? game->measureFailure() : 1;
    }

    virtual vector<UCT::SimAction*>& getActions () {
        return actVect;
    }

    virtual bool isTerminal () {
    	assert (game != NULL);
        return (game->fail() || game->terminate());
    }

    virtual void reset () {
    	assert (game != NULL);
        game->reset ();
        game->perturbation ();
        current->setState (game->getUCTState());
    }

    virtual void reset (net::ComputationGraph* policy) {
    	reset ();
    	this->policy = policy;
    }

    virtual void reset (DT* dt) {
        reset ();
        this->dt = dt;
    }
}; 

#endif /* GAME_H_ */