#ifndef __NEURAL_AGENT_H__
#define __NEURAL_AGENT_H__

#include <random>
#include <ctime>
#include <deque>
#include <cassert>
#include "chai.h"
#include "Params.h"

struct Memory{
	State s;
	double a;
	double r;
	State s_n;

	int fail;

	Memory(const State& s, double a, double r, const State& s_n, int fail)
	:s(s),a(a),r(r),s_n(s_n),fail(fail){
	}

	void checkout () {
		std::cout << "a: " << a << " r " << r << " fail: " << fail << "\n";
	}
};

class NeuralAgent : public TemporalDifferenceAgent {

private:
	//Net<N_IN,10,N_OUT> net; //subject to change
	//Net<N_IN,10,N_OUT> target_net; //target network

	ChaiModel nnmodel;

	StateActionPairTable<double> choices;

	std::deque<Memory> memories;
	int rSize; //recall size = # samples to recall
public:
	NeuralAgent(const bool test) //size of memory
		:TemporalDifferenceAgent("qlearning", test),
		//net(RHO, 1e-6, WEIGHT_DECAY),
		rSize(U_SIZE)
		
	{
		//net.copyTo(target_net); // same weights for starters
		nnmodel.AddFCL (4,10);
		nnmodel.AddSigmoid (10);
		nnmodel.AddFCL (10,10);
		nnmodel.AddSigmoid (10);
		nnmodel.AddFCL (10, 3);
		nnmodel.AddSigmoid (3);
	}

	virtual ~NeuralAgent() {

	}

	virtual void learn(const State & pre_state, int pre_action, double reward, const State & state, int) {
		memorize(pre_state, pre_action, reward, state);

		//learn_bundle (ALPHA, U_SIZE);
		//freeze ();
	}
	
	virtual void fail(const State & state, int action, double reward) {
		memorizeFailure(state, action, reward);

		learn_bundle (ALPHA, U_SIZE);
		freeze ();
	}

	double & qvalue(const State & state, const int & action)
	{
		//auto a = target_net.FF(state.vec());	
		//auto a = net.FF(state.vec());
		auto a = nnmodel.Evaluate (state.vec());
		float v = a[action+1];
		choices(state, action) = (double)v;
		return choices(state, action);
	}	

	std::pair<float,float> getBest(const State& state){
		//get best purely based on network feedforward q-value
		std::vector<float> v = state.vec();
		//currently editing here
		double maxVal = -99999;//reasonably small value
		double maxAction = 0.0;

		//auto a = target_net.FF(v);
		//auto a = net.FF(v);
		auto a = nnmodel.Evaluate (v);
		for(int i=0;i<3;++i){ 
			if(a[i] > maxVal){ //this is why R was favorite
				maxVal = a[i];
				maxAction = i-1;
			}
		}
		return std::make_pair(maxVal, maxAction);
	}

	void learn_policy (Memory& memory, double alpha, bool encourage) {
		static std::vector<float> y;
		const double a = memory.a;
		const double r = memory.r;

		auto s = memory.s.vec(); 
		
		y = nnmodel.Evaluate (s);
		if (encourage) {
			y[(int)a+1] = 0.9;
			for (int i = -1; i <= 1; i++) {
				if (i != a) y[i+1] = 0.1;
			}
		}
		
               else {
			y[(int)a+1] = 0.1;
			for (int i = -1; i <= 1; i++) {
				if (i != a) y[i+1] = 0.9;
			}
		}
		
		nnmodel.Train (s, y, 0.005);		
	}

	void learn(Memory& memory, double alpha){
		static std::vector<float> y;
		const double a = memory.a;
		const double r = memory.r;

		auto s = memory.s.vec(); 
		auto best = getBest(memory.s_n);
		auto maxqn = best.first;
		if (memory.fail) maxqn = 0;
		// stabilizing with target network
		//
		//y = target_net.FF(s); // compute target
		//y = net.FF(s);
		y = nnmodel.Evaluate (s);
		auto o = y[(int)a+1];
		if (memory.fail) {
			y[(int)a+1] = -1;
		}
               else
			y[(int)a+1] = (1-alpha)*y[(int)a+1] + (alpha)*(r+GAMMA*maxqn); //fill with new value
		auto n = y[(int)a+1];
		//o = net.FF(s)[(int)a+1]; // compuete output
		o = nnmodel.Evaluate(s)[(int)a+1];
		//net.BP(y); // back-propagate against target
		nnmodel.Train (s, y, 0.002);
		//cout << "maxqn = " << maxqn << endl;
		//n = net.FF(s)[(int)a+1];
		n = nnmodel.Evaluate(s)[(int)a+1];
		//std::cout << "trained from " << o << " to " << y[(int)a+1] << " result is "<< n << "!\n";
	}

	void learn_bundle(double alpha){
		// TODO : convert to minibatch
		static std::random_device rd;
		static std::mt19937 eng(rd());
		static std::uniform_int_distribution<int> distr(0,MEM_SIZE);

		for(int i=0;i<rSize;++i){
			learn(memories[distr(eng)], alpha);
		}
	}

	void learn_bundle(double alpha, int size){
		bool encourage = memories.size() >= 30;
		for(int i=0; i < memories.size();++i){
			int index = memories.size() - i - 1; //rand()%memories.size();
			//learn(memories[index],alpha);
			learn_policy (memories[index],alpha,encourage);
			//if (!encourage) {
			//	if (i > 10) break;
			//}
		}
		
		memories.clear ();
	}

	void memorize(const State& S, float a, double r, const State& next){	
		memories.emplace_back(S, a, r, next, 0);
		if(memories.size() > MEM_SIZE){
			memories.pop_front();
		}
	}

	void memorizeFailure(const State& S, float a, double r){		
		memories.emplace_back(S, a, r, S, 1);
		if(memories.size() > MEM_SIZE){
			memories.pop_front();
		}
	}

	void freeze(){
		//net.copyTo(target_net);
	}
};
#endif
