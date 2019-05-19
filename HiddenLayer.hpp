#ifndef HIDDEN_LAYER_HPP
#define HIDDEN_LAYER_HPP

class HiddenLayer
{
public:	

	int Inputs;
	int Outputs;
	
	HiddenLayer(int inputs, int outputs)
	{
		Inputs = inputs;
		Outputs = outputs;
	}
};
#endif
