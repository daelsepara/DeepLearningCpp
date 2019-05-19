#ifndef NEURAL_NETWORK_OPTIONS_HPP
#define NEURAL_NETWORK_OPTIONS_HPP

class NeuralNetworkOptions
{
public:

	double Alpha;
	int Epochs;
	int Inputs;
	int Nodes;
	int Items;
	int Categories;
	double Tolerance;
	int HiddenLayers;
	bool UseL2;

	NeuralNetworkOptions(double alpha, int epochs, int categories, int inputs, int nodes, int items, double tolerance)
	{
		Alpha = alpha;
		Epochs = epochs;
		Inputs = inputs; // Input layer features (i)
		Nodes = nodes; // Hidden layer nodes (j)
		Items = items;	// number of input items
		Categories = categories; // number of output categories (k)
		Tolerance = tolerance;
	}

	NeuralNetworkOptions(double alpha, int epochs, int categories, int inputs, int items, double tolerance, int hiddenLayers, bool useL2)
	{
		Alpha = alpha;
		Epochs = epochs;
		Inputs = inputs; // Input layer features (i)
		Items = items;	// number of input items
		Categories = categories; // number of output categories (k)
		Tolerance = tolerance;
		HiddenLayers = hiddenLayers; // number of hidden layers
		UseL2 = useL2;
	}

	NeuralNetworkOptions()
	{
		NeuralNetworkOptions(1.0, 1, 2, 2, 16, 50, 0.001);
	}
};

#endif

