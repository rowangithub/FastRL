#include "layer.hpp"
#include <iostream>
#include <vector>

namespace net
{
void ILayer::forward( const ComputationNode& input, ComputationNode& output ) const
{
	process( input.output(), output.out_cache() );
}

bool saveLayer (std::ofstream& file, Matrix& m) {
	if (!file.is_open())
	{
		std::cerr << "File is not open." << std::endl;
  		return false;
  	}

	file << std::fixed;
	file << m;
	file << "\nFIN";
	return true;
}

bool loadLayer (std::ifstream& input, Matrix& m) {
	if (input.fail()) {
		std::cerr << "ERROR. Cannot Read a fc_layer" << std::endl;
		return false;
	}

	std::string line;
	number_t d;
	std::vector<number_t> v;

	int n_rows = 0;
	while (getline(input, line))
	{
		if (line.compare ("FIN") == 0)
			break;
		++n_rows;
		std::stringstream input_line(line);
		while (!input_line.eof())
        	{
          		input_line >> d;
          		v.push_back(d);
        	}
      	}
	
	int n_cols = v.size()/n_rows;

	m.resize(n_rows, n_cols);
      
      	for (int i=0; i<n_rows; i++)
		for (int j=0; j<n_cols; j++)
         		m(i,j) = v[i*n_cols + j];
	return true;
}


bool saveLayerJson (std::vector<std::vector<std::vector<number_t>>>& params, Matrix& m) {
	std::vector<std::vector<number_t>> weights;
	for (int i = 0; i < m.rows(); i++) {
		std::vector<number_t> row;
		for (int j = 0; j < m.cols(); j++)
			row.push_back(m(i, j));
		weights.push_back(row);
	}
	params.push_back (weights);
	return true;
}
}
