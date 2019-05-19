#ifndef CONVOLUTIONAL_NEURAL_NETWORK_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_HPP

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "Convn.hpp"
#include "ConvolutionalNeuralNetworkOptions.hpp"
#include "File.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "MemOps.hpp"
#include "Profiler.hpp"
#include "Random.hpp"

class ConvolutionalNeuralNetwork
{
public:

	std::vector<Layer> Layers;

	double* Weights = NULL;
	double* WeightsDelta = NULL;
	double* WeightsTransposed = NULL;
	int WeightsX, WeightsY;

	double* FeatureVector = NULL;
	double* FeatureVectorDelta = NULL;
	int FeatureVectorLength;
	double* Output = NULL;
	double* OutputError = NULL;
	double* OutputDelta = NULL;
	int OutputX, OutputY;

	// 1D
	double* Bias = NULL;
	double* BiasDelta = NULL;
	int BiasLength;

	// Error
	double L = 0.0;

	std::vector<double> rL;

	ConvolutionalNeuralNetwork()
	{
		Weights = NULL;
		WeightsDelta = NULL;
		WeightsTransposed = NULL;
		FeatureVector = NULL;
		FeatureVectorDelta = NULL;
		Output = NULL;
		OutputError = NULL;
		OutputDelta = NULL;
		Bias = NULL;
		BiasDelta = NULL;
	}

	// Add Layer and Initialize
	void AddLayer(Layer layer)
	{
		if (layer.Type == LayerTypes::Input)
		{
			Layers.push_back(layer);
		}

		if (layer.Type == LayerTypes::Subsampling)
		{
			if (layer.Scale > 0)
			{
				Layers.push_back(layer);
			}
		}

		if (layer.Type == LayerTypes::Convolution)
		{
			if (layer.KernelSize > 0 && layer.OutputMaps > 0)
			{
				Layers.push_back(layer);
			}
		}
	}

	// Setup network input / output maps
	void Setup(int inputx, int inputy, int classes)
	{
		auto random = Random();
		random.UniformDistribution();

		auto InputMaps = 1;

		auto MapSizeX = inputx;
		auto MapSizeY = inputy;

		for (auto l = 0; l < (int)Layers.size(); l++)
		{
			if (Layers[l].Type == LayerTypes::Subsampling)
			{
				MapSizeX = MapSizeX / Layers[l].Scale;
				MapSizeY = MapSizeY / Layers[l].Scale;
			}

			if (Layers[l].Type == LayerTypes::Convolution)
			{
				MapSizeX = MapSizeX - Layers[l].KernelSize + 1;
				MapSizeY = MapSizeY - Layers[l].KernelSize + 1;

				Layers[l].FeatureMapLength = InputMaps * Layers[l].OutputMaps * Layers[l].KernelSize * Layers[l].KernelSize;
				Layers[l].FeatureMap = MemOps::New(Layers[l].FeatureMapLength);

				auto fan_out = Layers[l].OutputMaps * Layers[l].KernelSize * Layers[l].KernelSize;

				for (auto j = 0; j < Layers[l].OutputMaps; j++)
				{
					auto fan_in = InputMaps * Layers[l].KernelSize * Layers[l].KernelSize;

					for (auto i = 0; i < InputMaps; i++)
					{
						auto rand = MemOps::New(Layers[l].KernelSize * Layers[l].KernelSize);
						Rand(rand, random, Layers[l].KernelSize * Layers[l].KernelSize, fan_in, fan_out);
						MemOps::Copy2D4DIJ(Layers[l].FeatureMap, rand, InputMaps, Layers[l].OutputMaps, Layers[l].KernelSize, Layers[l].KernelSize, i, j);
						MemOps::Free(rand);
					}
				}

				Layers[l].Bias = MemOps::New(Layers[l].OutputMaps);
				MemOps::Set(Layers[l].Bias, Layers[l].OutputMaps, 0.0);
				
				InputMaps = Layers[l].OutputMaps;
			}
		}

		// 'features' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
		// 'classes' is the number of output neurons at the last layer, the layer just before the output layer.
		// 'Bias' is the biases of the output neurons.
		// 'Weights' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)

		auto features = MapSizeX * MapSizeY * InputMaps;

		Bias = MemOps::New(classes);
		MemOps::Set(Bias, classes, 0.0);
		BiasLength = classes;
		
		Weights = MemOps::New(features * classes);
		Rand(Weights, random, classes * features, features, classes);
		WeightsX = features;
		WeightsY = classes;
	}

	void Rand(double*& rand, Random random, int length, int fan_in, int fan_out)
	{
		for (auto x = 0; x < length; x++)
		{
			rand[x] = (random.NextDouble() - 0.5) * 2.0 * sqrt(6.0 / ((double)(fan_in + fan_out)));
		}
	}

	// Compute Forward Transform on 3D Input
	void FeedForward(double*& batch, int batchx, int batchy, int batchz, bool pool = false)
	{
		auto n = (int)Layers.size();

		auto InputMaps = 1;

		MemOps::Free(Layers[0].Activation);

		Layers[0].Activation = MemOps::New(InputMaps * batchx * batchy * batchz);
		Layers[0].ActivationX = batchx;
		Layers[0].ActivationY = batchy;
		Layers[0].ActivationZ = batchz;
		Layers[0].ActivationI = 1;

		MemOps::Copy4D3D(Layers[0].Activation, batch, batchx * batchy * batchz, 0);

		for (auto l = 1; l < n; l++)
		{
			if (Layers[l].Type == LayerTypes::Convolution)
			{
				auto zx = Layers[l - 1].ActivationX - Layers[l].KernelSize + 1;
				auto zy = Layers[l - 1].ActivationY - Layers[l].KernelSize + 1;
				auto zz = batchz;

				MemOps::Free(Layers[l].Activation);

				Layers[l].Activation = MemOps::New(Layers[l].OutputMaps * zx * zy * zz);
				Layers[l].ActivationX = zx;
				Layers[l].ActivationY = zy;
				Layers[l].ActivationZ = zz;
				Layers[l].ActivationI = Layers[l].OutputMaps;

				// !!below can probably be handled by insane matrix operations
				for (auto j = 0; j < Layers[l].OutputMaps; j++) // for each output map
				{
					// create temp output map
					auto z = MemOps::New(zx * zy * zz);

					for (auto i = 0; i < InputMaps; i++)
					{
						// copy Layers
						auto Activation = MemOps::New(Layers[l - 1].ActivationX * Layers[l - 1].ActivationY * batchz);
						auto FeatureMapX = Layers[l].KernelSize;
						auto FeatureMapY = Layers[l].KernelSize;
						auto FeatureMap = MemOps::New(FeatureMapX * FeatureMapY);

						MemOps::Copy4D3D(Activation, Layers[l - 1].Activation, Layers[l - 1].ActivationX * Layers[l - 1].ActivationY * batchz, i);
						MemOps::Copy4DIJ2D(FeatureMap, Layers[l].FeatureMap, FeatureMapX, FeatureMapY, Layers[l].OutputMaps, InputMaps, i, j);

						double* ztemp = NULL;

						// convolve with corresponding kernel and add to temp output map
						Convn::Valid(Activation, FeatureMap, ztemp, Layers[l - 1].ActivationX, Layers[l - 1].ActivationY, Layers[l - 1].ActivationZ, FeatureMapX, FeatureMapY, 1);
						Matrix::Add(z, ztemp, zx * zy * zz);

						MemOps::Free(ztemp);
						MemOps::Free(FeatureMap);
						MemOps::Free(Activation);
					}

					// add bias, pass through nonlinearity
					Matrix::Add(z, Layers[l].Bias[j], zx * zy * zz);
					auto sigm = Matrix::Sigm(z, zx * zy * zz);
					MemOps::Copy3D4D(Layers[l].Activation, sigm, zx * zy * zz, j);

					MemOps::Free(z);
					MemOps::Free(sigm);
				}

				InputMaps = Layers[l].OutputMaps;
			}
			else if (Layers[l].Type == LayerTypes::Subsampling)
			{
				// downsample

				// generate downsampling kernel
				auto FeatureMapX = Layers[l].Scale;
				auto FeatureMapY = Layers[l].Scale;
				auto FeatureMap = MemOps::New(FeatureMapX * FeatureMapY);
				auto scale = (double)Layers[l].Scale * Layers[l].Scale;

				for (auto i = 0; i < FeatureMapX * FeatureMapY; i++)
				{
					FeatureMap[i] = 1.0 / scale;
				}

				Layers[l].ActivationX = Layers[l - 1].ActivationX / Layers[l].Scale;
				Layers[l].ActivationY = Layers[l - 1].ActivationY / Layers[l].Scale;
				Layers[l].ActivationZ = batchz;
				Layers[l].ActivationI = InputMaps;

				MemOps::Free(Layers[l].Activation);

				Layers[l].Activation = MemOps::New(InputMaps * Layers[l].ActivationX * Layers[l].ActivationY * batchz);

				for (auto j = 0; j < InputMaps; j++)
				{
					// copy Layers
					auto ActivationZ = batchz;
					auto Activation = MemOps::New(Layers[l - 1].ActivationX * Layers[l - 1].ActivationY * batchz);

					MemOps::Copy4D3D(Activation, Layers[l - 1].Activation, Layers[l - 1].ActivationX * Layers[l - 1].ActivationY * batchz, j);

					auto zx = Layers[l - 1].ActivationX - FeatureMapX + 1;
					auto zy = Layers[l - 1].ActivationY - FeatureMapY + 1;
					auto zz = batchz;

					double* z = NULL;

					Convn::Valid(Activation, FeatureMap, z, Layers[l - 1].ActivationX, Layers[l - 1].ActivationY, batchz, FeatureMapX, FeatureMapY, 1);

					if (pool)
					{
						MemOps::Pool3D4D(Layers[l].Activation, z, Layers[l].ActivationX, Layers[l].ActivationY, Layers[l].ActivationZ, zx, zy, zz, j, Layers[l].Scale);
					}
					else
					{
						MemOps::Copy3D4D(Layers[l].Activation, z, Layers[l].ActivationX, Layers[l].ActivationY, Layers[l].ActivationZ, zx, zy, zz, j, Layers[l].Scale);
					}

					MemOps::Free(z);
					MemOps::Free(Activation);
				}

				MemOps::Free(FeatureMap);
			}
		}

		auto FeatureVectorX = Layers[n - 1].ActivationX;
		auto FeatureVectorY = Layers[n - 1].ActivationY;
		auto FeatureVectorN = Layers[n - 1].ActivationI;
		auto FeatureVectorZ = batchz;

		auto MapSize = FeatureVectorX * FeatureVectorY;
		auto FeatureSize = MapSize * FeatureVectorN;

		MemOps::Free(FeatureVector);

		FeatureVectorLength = FeatureSize * FeatureVectorZ;
		FeatureVector = MemOps::New(FeatureVectorLength);

		auto temp1D = MemOps::New(MapSize);
		auto temp2D = MemOps::New(MapSize);

		// concatenate all end layer feature maps into vector
		for (auto j = 0; j < FeatureVectorN; j++)
		{
			for (auto ii = 0; ii < FeatureVectorZ; ii++)
			{
				// Use Row-major in flattening the feature map
				MemOps::Copy4D2D(temp2D, Layers[n - 1].Activation, FeatureVectorX, FeatureVectorY, FeatureVectorX, FeatureVectorY, FeatureVectorZ, ii, j);
				Matrix::Transpose(temp1D, temp2D, FeatureVectorX, FeatureVectorY);
				MemOps::Copy2DOffset(FeatureVector, temp1D, FeatureVectorZ, FeatureSize, 1, MapSize, ii, j * MapSize);
			}
		}

		auto WeightsFeatureVector = MemOps::New(FeatureVectorZ * WeightsY);

		Matrix::Multiply(WeightsFeatureVector, Weights, FeatureVector, WeightsX, WeightsY, FeatureVectorZ, FeatureSize);

		auto repmat = MemOps::New(BiasLength * FeatureVectorZ);
		Matrix::Expand(Bias, 1, BiasLength, FeatureVectorZ, 1, repmat);
		Matrix::Add(WeightsFeatureVector, repmat, FeatureVectorZ * WeightsY);

		// feedforward into output perceptrons
		MemOps::Free(Output);

		Output = Matrix::Sigm(WeightsFeatureVector, FeatureVectorZ * WeightsY);
		OutputX = FeatureVectorZ;
		OutputY = WeightsY;

		MemOps::Free(WeightsFeatureVector);
		MemOps::Free(repmat);
		MemOps::Free(temp1D);
		MemOps::Free(temp2D);
	}

	// Classify data using trained network parameters and count classification errors
	int Classify(double*& test_input, double*& test_output, int testx, int testy, int testz, int classes, int items, int batchsize, double*& classification, bool pool = false)
	{
		auto errors = 0;

		auto tempx = MemOps::New(testx * testy * batchsize);
		auto tempy = MemOps::New(batchsize * classes);
		auto tempclass = MemOps::New(batchsize);

		MemOps::Free(classification);

		classification = MemOps::New(items);

		for (auto i = 0; i < items; i += batchsize)
		{
			// generate batch
			MemOps::Copy3D(tempx, test_input, testx, testy, batchsize, testx, testy, testz, 0, 0, i);
			MemOps::Copy2D(tempy, test_output, batchsize, classes, testz, classes, i, 0);

			// classify
			FeedForward(tempx, testx, testy, batchsize, pool);

			// count classifcation errors
			errors += Test(tempy, batchsize, classes, tempclass);

			// save classification
			MemOps::Copy2DOffset(classification, tempclass, items, 1, batchsize, 0, i, 0);
		}

		MemOps::Free(tempx);
		MemOps::Free(tempy);
		MemOps::Free(tempclass);

		return errors;
	}

	// Count classification errors
	int Test(double*& correct, int correctx, int correcty, double*& classifcation)
	{
		auto errors = 0;

		for (auto x = 0; x < OutputX; x++)
		{
			auto max = 0.0;
			auto cmax = 0.0;
			auto index = 0;
			auto cindex = 0;

			for (auto y = 0; y < OutputY; y++)
			{
				auto val = MemOps::Val2D(Output, OutputX, OutputY, x, y);

				if (val > max)
				{
					max = val;
					index = y;
				}
			}

			// Save classification
			classifcation[x] = index;

			for (auto cy = 0; cy < OutputY; cy++)
			{
				auto val = MemOps::Val2D(correct, correctx, correcty, x, cy);

				if (val > cmax)
				{
					cmax = val;
					cindex = cy;
				}
			}

			if (cindex != index)
				errors++;
		}

		return errors;
	}

	// Update Network Weights based on computed errors
	void BackPropagation(double*& batch, int batchx, int batchy)
	{
		auto n = (int)Layers.size();

		MemOps::Free(OutputDelta);
		MemOps::Free(OutputError);

		// backprop deltas
		OutputDelta = MemOps::New(batchx * batchy);
		OutputError = MemOps::New(batchx * batchy);

		for (auto x = 0; x < OutputX * OutputY; x++)
		{
			// error
			OutputError[x] = Output[x] - batch[x];

			// output delta
			OutputDelta[x] = OutputError[x] * (Output[x] * (1 - Output[x]));
		}

		// Loss Function
		L = 0.5 * Matrix::SquareSum(OutputError, OutputX * OutputY) / batchx;

		MemOps::Free(FeatureVectorDelta);
		MemOps::Free(WeightsTransposed);

		FeatureVectorDelta = MemOps::New(FeatureVectorLength);
		WeightsTransposed = MemOps::New(WeightsX * WeightsY);

		auto WeightsTransposedX = WeightsY;
		auto WeightsTransposedY = WeightsX;

		Matrix::Transpose(WeightsTransposed, Weights, WeightsX, WeightsY);

		// feature vector delta
		Matrix::Multiply(FeatureVectorDelta, WeightsTransposed, OutputDelta, WeightsTransposedX, WeightsTransposedY, batchx, batchy);

		// only conv layers has sigm function
		if (Layers[n - 1].Type == LayerTypes::Convolution)
		{
			for (auto x = 0; x < FeatureVectorLength; x++)
			{
				FeatureVectorDelta[x] *= FeatureVector[x] * (1 - FeatureVector[x]);
			}
		}

		// reshape feature vector deltas into output map style
		auto FeatureVectorX = Layers[n - 1].ActivationX;
		auto FeatureVectorY = Layers[n - 1].ActivationY;
		auto FeatureVectorN = Layers[n - 1].ActivationI;
		auto FeatureVectorZ = Layers[n - 1].ActivationZ;

		auto MapSize = FeatureVectorX * FeatureVectorY;
		auto FeatureSize = MapSize * FeatureVectorN;

		auto temp1D = MemOps::New(MapSize);
		auto temp2D = MemOps::New(MapSize);

		MemOps::Free(Layers[n - 1].Delta);

		Layers[n - 1].Delta = MemOps::New(FeatureSize * FeatureVectorZ);

		for (auto j = 0; j < FeatureVectorN; j++)
		{
			for (auto ii = 0; ii < FeatureVectorZ; ii++)
			{
				MemOps::Copy2D(temp1D, FeatureVectorDelta, 1, MapSize, FeatureVectorZ, FeatureSize, ii, j * MapSize);
				Matrix::Transpose(temp2D, temp1D, FeatureVectorX, FeatureVectorY);
				MemOps::Copy2D4D(Layers[n - 1].Delta, temp2D, FeatureVectorX, FeatureVectorY, FeatureVectorZ, FeatureVectorX, FeatureVectorY, ii, j);
			}
		}

		MemOps::Free(FeatureVectorDelta);
		MemOps::Free(temp1D);
		MemOps::Free(temp2D);

		for (auto l = n - 2; l >= 0; l--)
		{
			if (Layers[l].Type == LayerTypes::Convolution)
			{
				auto ActivationLength = Layers[l].ActivationX * Layers[l].ActivationY * Layers[l].ActivationZ * Layers[l].ActivationI;

				MemOps::Free(Layers[l].Delta);

				Layers[l].Delta = MemOps::New(ActivationLength);

				auto FeatureMapX = Layers[l + 1].ActivationX;
				auto FeatureMapY = Layers[l + 1].ActivationY;
				auto FeatureMapSize = FeatureMapX * FeatureMapY;

				auto FeatureMapExpandedLength = Layers[l + 1].Scale * Layers[l + 1].Scale * FeatureMapSize;

				auto FeatureMapExpanded = MemOps::New(FeatureMapExpandedLength);
				auto FeatureMap = MemOps::New(FeatureMapSize);

				for (auto j = 0; j < Layers[l].ActivationI; j++)
				{
					auto Activation = MemOps::New(FeatureMapExpandedLength);
					auto Delta = MemOps::New(FeatureMapExpandedLength);

					auto Scale = (1.0 / (double)(Layers[l + 1].Scale * Layers[l + 1].Scale));

					for (auto z = 0; z < FeatureVectorZ; z++)
					{
						MemOps::Copy4D2D(FeatureMap, Layers[l + 1].Delta, FeatureMapX, FeatureMapY, FeatureMapX, FeatureMapY, FeatureVectorZ, z, j);
						Matrix::Expand(FeatureMap, FeatureMapX, FeatureMapY, Layers[l + 1].Scale, Layers[l + 1].Scale, FeatureMapExpanded);
						MemOps::Copy4D2D(Activation, Layers[l].Activation, Layers[l].ActivationX, Layers[l].ActivationY, Layers[l].ActivationX, Layers[l].ActivationY, FeatureVectorZ, z, j);

						for (auto x = 0; x < FeatureMapExpandedLength; x++)
						{
							Delta[x] = Activation[x] * (1 - Activation[x]) * FeatureMapExpanded[x] * Scale;
						}

						MemOps::Copy2D4D(Layers[l].Delta, Delta, Layers[l].ActivationX, Layers[l].ActivationY, FeatureVectorZ, Layers[l].ActivationX, Layers[l].ActivationY, z, j);
					}

					MemOps::Free(Delta);
					MemOps::Free(Activation);
				}

				MemOps::Free(FeatureMapExpanded);
				MemOps::Free(FeatureMap);
			}
			else if (Layers[l].Type == LayerTypes::Subsampling)
			{
				auto sizei = Layers[l].ActivationI;
				auto sizej = Layers[l + 1].ActivationI;

				auto ActivationLength = Layers[l].ActivationX * Layers[l].ActivationY * Layers[l].ActivationZ * Layers[l].ActivationI;

				MemOps::Free(Layers[l].Delta);

				Layers[l].Delta = MemOps::New(ActivationLength);

				for (auto i = 0; i < sizei; i++)
				{
					auto zx = Layers[l].ActivationX;
					auto zy = Layers[l].ActivationY;
					auto zz = FeatureVectorZ;

					auto z = MemOps::New(zx * zy * zz);
					auto Delta = MemOps::New(Layers[l + 1].ActivationX * Layers[l + 1].ActivationY * FeatureVectorZ);

					for (auto j = 0; j < sizej; j++)
					{
						MemOps::Copy4D3D(Delta, Layers[l + 1].Delta, Layers[l + 1].ActivationX * Layers[l + 1].ActivationY * FeatureVectorZ, j);

						double* ztemp = NULL;
						double* rot180 = NULL;

						auto FeatureMapX = Layers[l + 1].KernelSize;
						auto FeatureMapY = Layers[l + 1].KernelSize;

						auto FeatureMap = MemOps::New(FeatureMapX * FeatureMapY);
						MemOps::Copy4DIJ2D(FeatureMap, Layers[l + 1].FeatureMap, FeatureMapX, FeatureMapY, sizej, sizei, i, j);
						Matrix::Rotate180(rot180, FeatureMap, FeatureMapX, FeatureMapY);
						Convn::Full(Delta, rot180, ztemp, Layers[l + 1].ActivationX, Layers[l + 1].ActivationY, FeatureVectorZ, FeatureMapX, FeatureMapY, 1);
						Matrix::Add(z, ztemp, zx * zy * zz);

						MemOps::Free(FeatureMap);
						MemOps::Free(rot180);
						MemOps::Free(ztemp);
					}

					MemOps::Copy3D4D(Layers[l].Delta, z, zx * zy * zz, i);

					MemOps::Free(Delta);
					MemOps::Free(z);
				}
			}
		}

		// calc gradients
		for (auto l = 1; l < n; l++)
		{
			if (Layers[l].Type == LayerTypes::Convolution)
			{
				MemOps::Free(Layers[l].DeltaFeatureMap);
				MemOps::Free(Layers[l].DeltaBias);

				Layers[l].DeltaFeatureMap = MemOps::New(Layers[l].FeatureMapLength);
				Layers[l].DeltaBias = MemOps::New(Layers[l].OutputMaps);

				for (auto j = 0; j < Layers[l].ActivationI; j++)
				{
					// d[j]
					auto dx = Layers[l].ActivationX;
					auto dy = Layers[l].ActivationY;
					auto dtemp = MemOps::New(dx * dy * FeatureVectorZ);

					MemOps::Copy4D3D(dtemp, Layers[l].Delta, dx * dy * FeatureVectorZ, j);

					for (auto i = 0; i < Layers[l - 1].ActivationI; i++)
					{
						// a[i]
						auto ax = Layers[l - 1].ActivationX;
						auto ay = Layers[l - 1].ActivationY;
						auto atemp = MemOps::New(ax * ay * FeatureVectorZ);

						double* ftemp = NULL;
						double* FeatureMapDelta = NULL;

						MemOps::Copy4D3D(atemp, Layers[l - 1].Activation, ax * ay * FeatureVectorZ, i);
						Matrix::FlipAll(ftemp, atemp, ax, ay, FeatureVectorZ);
						Convn::Valid(ftemp, dtemp, FeatureMapDelta, ax, ay, FeatureVectorZ, dx, dy, FeatureVectorZ);
						Matrix::Multiply(FeatureMapDelta, Layers[l].KernelSize * Layers[l].KernelSize, 1.0 / (double)FeatureVectorZ);
						MemOps::Copy2D4DIJ(Layers[l].DeltaFeatureMap, FeatureMapDelta, Layers[l - 1].ActivationI, Layers[l].ActivationI, Layers[l].KernelSize, Layers[l].KernelSize, i, j);

						MemOps::Free(FeatureMapDelta);
						MemOps::Free(ftemp);
						MemOps::Free(atemp);
					}

					Layers[l].DeltaBias[j] = Matrix::Sum(dtemp, dx * dy * FeatureVectorZ) / FeatureVectorZ;

					MemOps::Free(dtemp);
				}
			}
		}

		MemOps::Free(WeightsDelta);

		WeightsDelta = MemOps::New(WeightsX * WeightsY);

		auto FeatureVectorTransposed = MemOps::New(FeatureVectorLength);

		Matrix::Transpose(FeatureVectorTransposed, FeatureVector, FeatureVectorZ, FeatureSize);
		Matrix::Multiply(WeightsDelta, OutputDelta, FeatureVectorTransposed, batchx, batchy, FeatureSize, FeatureVectorZ);
		Matrix::Multiply(WeightsDelta, WeightsX * WeightsY, 1.0 / (double)FeatureVectorZ);
		Matrix::Mean(BiasDelta, OutputDelta, batchx, batchy, 0);

		MemOps::Free(WeightsTransposed);
		MemOps::Free(FeatureVectorTransposed);
		MemOps::Free(FeatureVector);
		MemOps::Free(OutputDelta);
		MemOps::Free(OutputError);
	}

	void ApplyGradients(ConvolutionalNeuralNetworkOptions opts)
	{
		for (auto l = 1; l < (int)Layers.size(); l++)
		{
			if (Layers[l].Type == LayerTypes::Convolution)
			{
				auto length = Layers[l].KernelSize * Layers[l].KernelSize * Layers[l - 1].ActivationI * Layers[l].ActivationI;

				Matrix::Add(Layers[l].FeatureMap, Layers[l].DeltaFeatureMap, length, -opts.Alpha);
				Matrix::Add(Layers[l].Bias, Layers[l].DeltaBias, Layers[l].OutputMaps, -opts.Alpha);

				MemOps::Free(Layers[l].DeltaBias);
				MemOps::Free(Layers[l].DeltaFeatureMap);
			}

			MemOps::Free(Layers[l].Delta);
		}

		Matrix::Add(Weights, WeightsDelta, WeightsX * WeightsY, -opts.Alpha);
		Matrix::Add(Bias, BiasDelta, BiasLength, -opts.Alpha);

		MemOps::Free(WeightsDelta);
		MemOps::Free(BiasDelta);
	}

	void Train(double*& input, int inputx, int inputy, int inputz, double*& output, int outputx, int outputy, ConvolutionalNeuralNetworkOptions opts)
	{
		auto index_list = MemOps::IntList(opts.Items);

		for (auto epoch = 0; epoch < opts.Epochs; epoch++)
		{
			if (opts.Shuffle)
			{
				auto seed = static_cast<unsigned int>(time(0));

				srand(seed);

				std::random_shuffle(&index_list[0], &index_list[opts.Items]);
			}

			auto temp_input = MemOps::New(inputx * inputy * opts.BatchSize);
			auto temp_output = MemOps::New(opts.BatchSize * outputy);

			auto start = Profiler::now();

			auto batchnum = 0;

			auto rLVal = 0.0;

			rL.clear();

			for (auto i = 0; i < opts.Items; i += opts.BatchSize)
			{
				if (opts.Shuffle)
				{
					MemOps::Copy3DZ(temp_input, input, index_list, inputx, inputy, opts.BatchSize, inputx, inputy, inputz, i);
					MemOps::Copy2DX(temp_output, output, index_list, opts.BatchSize, outputy, outputx, outputy, i);
				}
				else
				{
					MemOps::Copy3D(temp_input, input, inputx, inputy, opts.BatchSize, inputx, inputy, inputz, 0, 0, i);
					MemOps::Copy2D(temp_output, output, opts.BatchSize, outputy, outputx, outputy, i, 0);
				}

				FeedForward(temp_input, inputx, inputy, opts.BatchSize, opts.Pool);
				BackPropagation(temp_output, opts.BatchSize, outputy);
				ApplyGradients(opts);

				if (rL.size() == 0)
				{
					rL.push_back(L);
				}

				rLVal = 0.99 * rL[rL.size() - 1] + 0.01 * L;

				rL.push_back(rLVal);

				batchnum++;
			}

			printf("epoch %d/%d elapsed time is %ld ms - Error: %f\n", epoch + 1, opts.Epochs, Profiler::Elapsed(start), rLVal);

			MemOps::Free(temp_input);
			MemOps::Free(temp_output);
		}

		MemOps::Free(index_list);
	}

	void Free()
	{
		for (auto i = 0; i < (int)Layers.size(); i++)
		{
			MemOps::Free(Layers[i].Activation);

			if (Layers[i].Type == LayerTypes::Convolution)
			{
				MemOps::Free(Layers[i].FeatureMap);
				MemOps::Free(Layers[i].Bias);
			}
		}

		MemOps::Free(Weights);
		MemOps::Free(Output);
		MemOps::Free(Bias);

		rL.clear();
		Layers.clear();
	}
};

#endif
