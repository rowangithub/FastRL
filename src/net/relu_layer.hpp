
#pragma once

#include "layer.hpp"

namespace net
{
class ReLULayer : public ILayer
{
public:
	explicit ReLULayer(Matrix p) : mBias(p) {};

	/// get the size of the layer output
	std::size_t getOutputSize() const override;
	
	const Matrix& getParameter() const { return mBias; };

	// propagates input forward and calculates output
	void process(const Vector& input, Vector& out) const override;

	// propagates error backward, and uses solver to track gradient
	void backward(const Vector& error, Vector& back, const ComputationNode& compute, Solver& solver) const override;

	void update(Solver& solver) override;
	
	std::unique_ptr<ILayer> clone() const override;

	bool save (std::ofstream& file) override;
	bool load (std::ifstream& input) override;
	bool saveJson (std::vector<std::vector<std::vector<number_t>>>& weights, std::vector<std::vector<std::vector<number_t>>>& biases) override;
private:
	Matrix mBias;
};
}
