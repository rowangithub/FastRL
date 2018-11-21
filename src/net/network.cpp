#include "network.hpp"
#include "fc_layer.hpp"
#include "relu_layer.hpp"
#include "tanh_layer.hpp"

#include <iostream>
#include <regex>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

// Short alias for this namespace
namespace pt = boost::property_tree;

namespace net
{
Network& Network::add_layer_imp( layer_t layer )
{
	mLayers.push_back( std::move(layer) );
	return *this;
}

/*ComputationNode Network::forward(Vector input) const
{
	ComputationNode node( std::move(input) );

	for(auto& layer : mLayers)
	{
		node = layer->forward( std::move(node) );
	}
	return node;
}


ComputationNode Network::operator()( Vector input ) const
{
	return forward( std::move(input) );
}
*/
void Network::update(Solver& solver)
{
    for(auto& l : mLayers)
	{
        l->update(solver);
	}
}

Network Network::clone() const
{
	Network newnet;
	for(const auto& layer : mLayers)
	{
		newnet.mLayers.push_back( layer->clone() );
	}
	return newnet;
}

bool Network::save (std::string filename)
{     
	std::ofstream file;

	file.open(filename.c_str());

	for (layer_t layer : mLayers) {
		layer->save (file);	
		file << "\n";	
	}
	file << "END";

	file.close ();
	return true;
}

bool Network::load (std::string filename)
{
	std::ifstream input(filename.c_str());
	if (input.fail()) {
		std::cerr << "ERROR. Cannot Read a fc_layer" << std::endl;
		return false;
	}

	std::string line;
	while (getline(input, line))
	{
		Matrix m;
		if (line.compare ("FC") == 0) {
			loadLayer (input, m);
			(*this) << FcLayer (m);
		} 
		else if (line.compare ("RELU") == 0) {
			loadLayer (input, m);
			(*this) << ReLULayer (m);
		}
		else if (line.compare ("TANH") == 0) {
			loadLayer (input, m);
			(*this) << TanhLayer (m);
		}
		else if (line.compare ("END") == 0) {
			break;
		}
		else {
			input.close ();
			return false;
		}
	}
	input.close ();
	return true;
}

std::string fix_json_numbers(const std::string& json_str)
{
    std::regex re("\\\"(-?[0-9]+\\.{0,1}[0-9]*(e-?[0-9]+)?)\\\"");
    return std::regex_replace(json_str, re, "$1");
}

bool Network::saveJson (std::string filename)
{
	
	std::vector<std::vector<std::vector<number_t>>> weights;
	std::vector<std::vector<std::vector<number_t>>> biases;

	for (layer_t layer : mLayers) {
		layer->saveJson(weights, biases);	
	}

	pt::ptree root;

	root.put("cost", "CrossEntropyCost");
	pt::ptree content;
	for (int i = 0; i < weights.size(); i++)
	{
    	pt::ptree matrix_node;
    	const std::vector<std::vector<number_t>>& matrix = weights[i];

    	for (int j = 0; j < matrix.size(); j++)
    	{
    		pt::ptree row;
    		const std::vector<number_t> matrix_row = matrix[j];
    		for (int k = 0; k < matrix_row.size(); k++) {
        		pt::ptree cell;
        		cell.put_value(matrix_row[k]);
        		row.push_back(std::make_pair("", cell));
        	}
    		matrix_node.push_back(std::make_pair("", row));
    	}
    	content.push_back(std::make_pair("", matrix_node));
	}
	root.add_child("weights", content);


	pt::ptree content2;
	for (int i = 0; i < biases.size(); i++)
	{
    	pt::ptree matrix_node;
    	const std::vector<std::vector<number_t>>& matrix = biases[i];

    	for (int j = 0; j < matrix.size(); j++)
    	{
    		pt::ptree row;
    		const std::vector<number_t> matrix_row = matrix[j];
    		for (int k = 0; k < matrix_row.size(); k++) {
        		pt::ptree cell;
        		cell.put_value(matrix_row[k]);
        		row.push_back(std::make_pair("", cell));
        	}
    		matrix_node.push_back(std::make_pair("", row));
    	}
    	content2.push_back(std::make_pair("", matrix_node));
	}
	root.add_child("biases", content2);

	pt::ptree size;
	std::vector<int> sizes;

	int pre_m = -1, pre_n = -1, m, n;
	for (int i = 0; i < weights.size(); i++) {
		n = weights[i].size();
		m = weights[i][0].size();
		if (pre_n != m) 
			sizes.push_back(m);
		sizes.push_back(n);

		pre_m = m;
		pre_n = n;
	}

	for (int s : sizes) {
    	pt::ptree size_node;
    	size_node.put("", s);

    	size.push_back(std::make_pair("", size_node));
	}
	root.add_child("sizes", size);

	std::ostringstream buf; 
	pt::write_json (buf, root);
	std::string treetext = fix_json_numbers(buf.str());

	std::ofstream file;
	file.open(filename.c_str());
	file << treetext;
	file.close();

	return true;
}
}
