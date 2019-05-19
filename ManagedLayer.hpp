#ifndef MANAGED_LAYER_HPP
#define MANAGED_LAYER_HPP

#include "LayerTypes.hpp"
#include "ManagedArray.hpp"

class ManagedLayer
{
public:

	LayerTypes Type;

	int OutputMaps = 0;
	int Scale = 0;
	int KernelSize = 0;

	ManagedArray FeatureMap = NULL;     // FeatureMap[i][j][x][y]
	ManagedArray DeltaFeatureMap = NULL;

	ManagedArray Activation = NULL; // Activation[i][x][y][z]
	ManagedArray Delta = NULL;

	// 1D
	ManagedArray Bias = NULL;
	ManagedArray DeltaBias = NULL;

	ManagedLayer()
	{
		Type = Input;
		Activation = NULL;
		Delta = NULL;
		Bias = NULL;
		DeltaBias = NULL;
		FeatureMap = NULL;
		DeltaFeatureMap = NULL;
	}

	ManagedLayer(int scale)
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

	ManagedLayer(int outputMaps, int kernelSize)
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
