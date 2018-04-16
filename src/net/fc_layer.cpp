#include "fc_layer.hpp"
#include "solver.hpp"

namespace net
{
/// get the size of the layer output
std::size_t FcLayer::getOutputSize() const
{
	return mMatrix.rows();
}
	
void FcLayer::process(const Vector& input, Vector& out) const
{
	out.noalias() = mMatrix * input;
}

void FcLayer::backward(const Vector& error, Vector& back, const ComputationNode& compute, Solver& solver) const
{
	solver(mMatrix, error * compute.input().transpose());
	back.noalias() = mMatrix.transpose() * error;
}

void FcLayer::update(Solver& solver)
{
	solver.update( mMatrix );
}

std::unique_ptr<ILayer> FcLayer::clone() const
{
	return std::make_unique<FcLayer>( *this );
}

bool FcLayer::save (std::ofstream& file) {
	file << "FC\n"; 
	return saveLayer (file, mMatrix);
}

bool FcLayer::load (std::ifstream& input) {
	return loadLayer (input, mMatrix);
}

bool FcLayer::saveJson (std::vector<std::vector<std::vector<number_t>>>& weights,
						std::vector<std::vector<std::vector<number_t>>>& biases) {
	return saveLayerJson (weights, mMatrix);
}
}
