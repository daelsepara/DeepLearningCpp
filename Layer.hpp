#ifndef LAYER_HPP
#define LAYER_HPP

#include "LayerTypes.hpp"

class Layer
{
public:

	LayerTypes Type;
	int OutputMaps = 0;
	int Scale = 0;
	int KernelSize = 0;

	double* FeatureMap = NULL;     // FeatureMap[i][j][x][y]
	double* DeltaFeatureMap = NULL;
	int FeatureMapLength;

	double* Activation = NULL; // Activation[i][x][y][z]
	double* Delta = NULL;
	int ActivationI, ActivationX, ActivationY, ActivationZ;

	// 1D
	double* Bias = NULL;
	double* DeltaBias = NULL;

	Layer()
	{
		Type = Input;
		Activation = NULL;
		Delta = NULL;
		Bias = NULL;
		DeltaBias = NULL;
		FeatureMap = NULL;
		DeltaFeatureMap = NULL;
	}

	Layer(int scale)
	{
		Type = Subsampling;
		Scale = scale;
		Activation = NULL;
		Delta = NULL;
		Bias = NULL;
		DeltaBias = NULL;
		FeatureMap = NULL;
		DeltaFeatureMap = NULL;
	}

	Layer(int outputMaps, int kernelSize)
	{
		Type = Convolution;
		KernelSize = kernelSize;
		OutputMaps = outputMaps;
		Activation = NULL;
		Delta = NULL;
		Bias = NULL;
		DeltaBias = NULL;
		FeatureMap = NULL;
		DeltaFeatureMap = NULL;
	}
};

#endif
