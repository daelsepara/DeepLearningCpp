#ifndef CONVOLUTIONAL_NEURAL_NETWORK_OPTIONS_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_OPTIONS_HPP

class ConvolutionalNeuralNetworkOptions
{
public:

	double Alpha;
	int BatchSize;
	int Epochs;
	int Items;
	bool Pool;
	bool Shuffle;

	int InputX;
	int InputY;
	int Classes;

	ConvolutionalNeuralNetworkOptions(double alpha, int batchsize, int epochs, int items, bool pool = false, bool shuffle = false)
	{
		Alpha = alpha;
		BatchSize = batchsize;
		Epochs = epochs;
		Items = items;
		Pool = pool;
		Shuffle = shuffle;
	}

	ConvolutionalNeuralNetworkOptions()
	{
		ConvolutionalNeuralNetworkOptions(1.0, 50, 1, 50, false, false);
	}

	void SetInputSize(int inputx, int inputy)
	{
		InputX = inputx;
		InputY = inputy;
	}

	void SetClasses(int classes)
	{
		Classes = classes;
	}
};

#endif
