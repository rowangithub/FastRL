#include "action.h"
#include "computation_graph.hpp"

namespace qlearn
{
	using namespace net;
	Action getAction(ComputationGraph& graph, const Vector& situation)
	{
		const auto& result = graph.forward( situation );
		// greedy algorithm that generates the next action.
		int row, col;
		float quality = result.maxCoeff(&row,&col);
		return {(std::size_t)row, quality};
	}
}
