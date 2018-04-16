#include "tanh_layer.hpp"
#include "solver.hpp"
#include <cmath>

namespace net
{
/// get the size of the layer output
std::size_t TanhLayer::getOutputSize() const
{
	return mBias.size();
}

void TanhLayer::process(const Vector& input, Vector& out) const
{
	out = (mBias + input).unaryExpr([](float x) { return std::tanh(x);} );
}

void TanhLayer::backward(const Vector& error, Vector& back, const ComputationNode& compute, Solver& solver) const
{
	auto deriv = [](number_t v) { return 1 - v*v; };
	back =  error.array() * (compute.output().unaryExpr(deriv)).array();
	solver(mBias, back);
}

void TanhLayer::update(Solver& solver)
{
	solver.update( mBias );
}

std::unique_ptr<ILayer> TanhLayer::clone() const
{
	return std::unique_ptr<ILayer>( new TanhLayer(*this) );
}

bool TanhLayer::save (std::ofstream& file) {
	file << "TANH\n";
	return saveLayer (file, mBias);
}

bool TanhLayer::load (std::ifstream& input) {
	return loadLayer (input, mBias);
}

bool TanhLayer::saveJson (std::vector<std::vector<std::vector<number_t>>>& weights,
						std::vector<std::vector<std::vector<number_t>>>& biases) {
	return saveLayerJson (biases, mBias);
}
}
