#ifndef MANAGED_CNN_HPP
#define MANAGED_CNN_HPP

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ConvolutionalNeuralNetworkOptions.hpp"
#include "ManagedArray.hpp"
#include "ManagedConvolution.hpp"
#include "ManagedFile.hpp"
#include "ManagedLayer.hpp"
#include "ManagedMatrix.hpp"
#include "ManagedOps.hpp"
#include "Profiler.hpp"
#include "Random.hpp"

class ManagedCNN
{
public:

	std::vector<ManagedLayer> Layers;

	ManagedArray Weights = NULL;
	ManagedArray WeightsDelta = NULL;

	ManagedArray FeatureVector = NULL;

	ManagedArray Output = NULL;
	ManagedArray OutputError = NULL;
	ManagedArray OutputDelta = NULL;

	// 1D
	ManagedArray Bias = NULL;
	ManagedArray BiasDelta = NULL;

	// Error
	double L = 0.0;

	std::vector<double> rL;

	ManagedCNN()
	{

	}

	// Add Layer and Initialize
	void AddLayer(ManagedLayer layer)
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
	void Setup(ManagedArray input, int classes)
	{
		auto random = Random();
		random.UniformDistribution();

		auto InputMaps = 1;

		auto MapSizeX = input.x;
		auto MapSizeY = input.y;

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

				Layers[l].FeatureMap.Resize(Layers[l].KernelSize, Layers[l].KernelSize, 1, InputMaps, Layers[l].OutputMaps);

				auto fan_out = Layers[l].OutputMaps * Layers[l].KernelSize * Layers[l].KernelSize;

				for (auto j = 0; j < Layers[l].OutputMaps; j++)
				{
					auto fan_in = InputMaps * Layers[l].KernelSize * Layers[l].KernelSize;

					for (auto i = 0; i < InputMaps; i++)
					{
						auto rand = ManagedArray(Layers[l].KernelSize, Layers[l].KernelSize);
						Rand(rand, random, fan_in, fan_out);
						ManagedOps::Copy2D4DIJ(Layers[l].FeatureMap, rand, i, j);
						ManagedOps::Free(rand);
					}
				}

				Layers[l].Bias.Resize(Layers[l].OutputMaps);
				ManagedOps::Set(Layers[l].Bias, 0.0);
				
				InputMaps = Layers[l].OutputMaps;
			}
		}

		// 'classes' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
		// 'features' is the number of output neurons at the last layer, the layer just before the output layer.
		// 'Bias' is the biases of the output neurons.
		// 'Weights' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)

		auto features = MapSizeX * MapSizeY * InputMaps;

		Bias.Resize(1, classes);
		ManagedOps::Set(Bias, 0.0);
		
		Weights.Resize(features, classes);
		Rand(Weights, random, features, classes);
	}

	void Rand(ManagedArray& rand, Random random, int fan_in, int fan_out)
	{
		for (auto x = 0; x < rand.Length(); x++)
		{
			rand(x) = (random.NextDouble() - 0.5) * 2.0 * sqrt(6.0 / ((double)(fan_in + fan_out)));
		}
	}

	// Compute Forward Transform on 3D Input
	void FeedForward(ManagedArray& batch, bool pool = false)
	{
		auto n = (int)Layers.size();

		auto InputMaps = 1;

		Layers[0].Activation.Resize(batch, false);

		ManagedOps::Copy4D3D(Layers[0].Activation, batch, 0);

		for (auto l = 1; l < n; l++)
		{
			if (Layers[l].Type == LayerTypes::Convolution)
			{
				auto zx = Layers[l - 1].Activation.x - Layers[l].KernelSize + 1;
				auto zy = Layers[l - 1].Activation.y - Layers[l].KernelSize + 1;
				auto zz = batch.z;

				Layers[l].Activation.Resize(zx, zy, zz, Layers[l].OutputMaps, 1, false);

				auto FeatureMapX = Layers[l].KernelSize;
				auto FeatureMapY = Layers[l].KernelSize;

				auto Activation = ManagedArray(Layers[l - 1].Activation.x, Layers[l - 1].Activation.y, batch.z, false);
				auto FeatureMap = ManagedArray(FeatureMapX, FeatureMapY, false);

				ManagedArray ztemp;

				// !!below can probably be handled by insane matrix operations
				for (auto j = 0; j < Layers[l].OutputMaps; j++) // for each output map
				{
					// create temp output map
					auto z = ManagedArray(zx, zy, zz);

					for (auto i = 0; i < InputMaps; i++)
					{
						// copy Layers
						ManagedOps::Copy4D3D(Activation, Layers[l - 1].Activation, i);
						ManagedOps::Copy4DIJ2D(FeatureMap, Layers[l].FeatureMap, i, j);

						// convolve with corresponding kernel and add to temp output map
						ManagedConvolution::Valid(Activation, FeatureMap, ztemp);
						ManagedMatrix::Add(z, ztemp);
					}

					// add bias, pass through nonlinearity
					ManagedMatrix::Add(z, Layers[l].Bias(j));
					auto sigm = ManagedMatrix::Sigm(z);
					ManagedOps::Copy3D4D(Layers[l].Activation, sigm, j);

					ManagedOps::Free(sigm);
					ManagedOps::Free(z);
				}

				ManagedOps::Free(Activation);
				ManagedOps::Free(FeatureMap);
				ManagedOps::Free(ztemp);

				InputMaps = Layers[l].OutputMaps;
			}
			else if (Layers[l].Type == LayerTypes::Subsampling)
			{
				// downsample

				// generate downsampling kernel
				auto FeatureMapX = Layers[l].Scale;
				auto FeatureMapY = Layers[l].Scale;
				auto scale = (double)(Layers[l].Scale * Layers[l].Scale);

				auto FeatureMap = ManagedArray(FeatureMapX, FeatureMapY, false);
				ManagedOps::Set(FeatureMap, 1.0 / scale);

				Layers[l].Activation.Resize(Layers[l - 1].Activation.x / Layers[l].Scale, Layers[l - 1].Activation.y / Layers[l].Scale, batch.z, InputMaps, 1, false);

				auto Activation = ManagedArray(Layers[l - 1].Activation.x, Layers[l - 1].Activation.y, batch.z, false);

				for (auto j = 0; j < InputMaps; j++)
				{
					ManagedArray z;

					// copy Layers
					ManagedOps::Copy4D3D(Activation, Layers[l - 1].Activation, j);

					// Subsample
					ManagedConvolution::Valid(Activation, FeatureMap, z);

					if (pool)
					{
						ManagedOps::Pool3D4D(Layers[l].Activation, z, j, Layers[l].Scale);
					}
					else
					{
						ManagedOps::Copy3D4D(Layers[l].Activation, z, j, Layers[l].Scale);
					}

					ManagedOps::Free(z);
				}

				ManagedOps::Free(Activation);
				ManagedOps::Free(FeatureMap);
			}
		}

		auto FeatureVectorX = Layers[n - 1].Activation.x;
		auto FeatureVectorY = Layers[n - 1].Activation.y;
		auto FeatureVectorN = Layers[n - 1].Activation.i;
		auto FeatureVectorZ = batch.z;

		auto MapSize = FeatureVectorX * FeatureVectorY;
		auto FeatureSize = MapSize * FeatureVectorN;

		FeatureVector.Resize(FeatureVectorZ, FeatureSize, false);

		ManagedArray temp1D;

		auto temp2D = ManagedArray(FeatureVectorX, FeatureVectorY, false);

		// concatenate all end layer feature maps into vector
		for (auto j = 0; j < FeatureVectorN; j++)
		{
			for (auto ii = 0; ii < FeatureVectorZ; ii++)
			{
				// Use Row-major in flattening the feature map
				ManagedOps::Copy4D2D(temp2D, Layers[n - 1].Activation, ii, j);
				ManagedMatrix::Transpose(temp1D, temp2D);
				temp1D.Reshape(1, MapSize);
				ManagedOps::Copy2DOffset(FeatureVector, temp1D, ii, j * MapSize);
			}
		}

		auto WeightsFeatureVector = ManagedMatrix::Multiply(Weights, FeatureVector);
		auto repmat = ManagedArray(FeatureVectorZ, Bias.Length(), false);
		ManagedMatrix::Expand(Bias, FeatureVectorZ, 1, repmat);
		ManagedMatrix::Add(WeightsFeatureVector, repmat);

		ManagedOps::Free(Output);

		// feedforward into output perceptrons
		Output = ManagedMatrix::Sigm(WeightsFeatureVector);

		ManagedOps::Free(WeightsFeatureVector);
		ManagedOps::Free(repmat);
		ManagedOps::Free(temp1D);
		ManagedOps::Free(temp2D);
	}

	// Update Network Weights based on computed errors
	void BackPropagation(ManagedArray& batch)
	{
		auto n = (int)Layers.size();

		// backprop deltas
		OutputDelta.Resize(batch, false);
		OutputError.Resize(batch, false);

		for (auto x = 0; x < Output.Length(); x++)
		{
			// error
			OutputError(x) = Output(x) - batch(x);

			// output delta
			OutputDelta(x) = OutputError(x) * (Output(x) * (1 - Output(x)));
		}

		// Loss Function
		L = 0.5 * ManagedMatrix::SquareSum(OutputError) / batch.x;

		// feature vector delta
		auto WeightsTransposed = ManagedMatrix::Transpose(Weights);
		auto FeatureVectorDelta = ManagedMatrix::Multiply(WeightsTransposed, OutputDelta);

		// only conv layers has sigm function
		if (Layers[n - 1].Type == LayerTypes::Convolution)
		{
			for (auto x = 0; x < FeatureVectorDelta.Length(); x++)
			{
				FeatureVectorDelta(x) = FeatureVectorDelta(x) * FeatureVector(x) * (1 - FeatureVector(x));
			}
		}

		// reshape feature vector deltas into output map style
		auto FeatureVectorX = Layers[n - 1].Activation.x;
		auto FeatureVectorY = Layers[n - 1].Activation.y;
		auto FeatureVectorN = Layers[n - 1].Activation.i;
		auto FeatureVectorZ = Layers[n - 1].Activation.z;

		auto MapSize = FeatureVectorX * FeatureVectorY;

		auto temp1D = ManagedArray(1, MapSize, false);

		ManagedArray temp2D;

		Layers[n - 1].Delta.Resize(FeatureVectorX, FeatureVectorY, FeatureVectorZ, FeatureVectorN, 1, false);

		for (auto j = 0; j < FeatureVectorN; j++)
		{
			for (auto ii = 0; ii < FeatureVectorZ; ii++)
			{
				ManagedOps::Copy2D(temp1D, FeatureVectorDelta, ii, j * MapSize);
				temp1D.Reshape(FeatureVectorX, FeatureVectorY);
				ManagedMatrix::Transpose(temp2D, temp1D);
				ManagedOps::Copy2D4D(Layers[n - 1].Delta, temp2D, ii, j);
				temp1D.Reshape(1, MapSize);
			}
		}

		ManagedOps::Free(FeatureVectorDelta);
		ManagedOps::Free(temp1D);
		ManagedOps::Free(temp2D);

		for (auto l = n - 2; l >= 0; l--)
		{
			if (Layers[l].Type == LayerTypes::Convolution)
			{
				Layers[l].Delta.Resize(Layers[l].Activation, false);

				auto FeatureMapX = Layers[l + 1].Activation.x;
				auto FeatureMapY = Layers[l + 1].Activation.y;
				auto xx = Layers[l + 1].Scale * FeatureMapX;
				auto yy = Layers[l + 1].Scale * FeatureMapY;

				auto FeatureMapExpanded = ManagedArray(xx, yy, false);
				auto FeatureMap = ManagedArray(FeatureMapX, FeatureMapY, false);

				for (auto j = 0; j < Layers[l].Activation.i; j++)
				{
					auto Activation = ManagedArray(xx, yy, false);
					auto Delta = ManagedArray(xx, yy, false);

					auto Scale = (1.0 / (double)(Layers[l + 1].Scale * Layers[l + 1].Scale));

					for (auto z = 0; z < FeatureVectorZ; z++)
					{
						ManagedOps::Copy4D2D(FeatureMap, Layers[l + 1].Delta, z, j);
						ManagedMatrix::Expand(FeatureMap, Layers[l + 1].Scale, Layers[l + 1].Scale, FeatureMapExpanded);
						ManagedOps::Copy4D2D(Activation, Layers[l].Activation, z, j);

						for (auto x = 0; x < Delta.Length(); x++)
						{
							Delta(x) = Activation(x) * (1 - Activation(x)) * FeatureMapExpanded(x) * Scale;
						}

						ManagedOps::Copy2D4D(Layers[l].Delta, Delta, z, j);
					}

					ManagedOps::Free(Delta);
					ManagedOps::Free(Activation);
				}

				ManagedOps::Free(FeatureMapExpanded);
				ManagedOps::Free(FeatureMap);
			}
			else if (Layers[l].Type == LayerTypes::Subsampling)
			{
				auto sizei = Layers[l].Activation.i;
				auto sizej = Layers[l + 1].Activation.i;

				Layers[l].Delta.Resize(Layers[l].Activation, false);

				for (auto i = 0; i < sizei; i++)
				{
					auto zx = Layers[l].Activation.x;
					auto zy = Layers[l].Activation.y;
					auto zz = FeatureVectorZ;

					auto z = ManagedArray(zx, zy, zz);
					auto Delta = ManagedArray(Layers[l + 1].Activation.x, Layers[l + 1].Activation.y, FeatureVectorZ, false);

					for (auto j = 0; j < sizej; j++)
					{
						ManagedArray ztemp, rot180;

						auto FeatureMapX = Layers[l + 1].KernelSize;
						auto FeatureMapY = Layers[l + 1].KernelSize;

						auto FeatureMap = ManagedArray(FeatureMapX, FeatureMapY, false);
						ManagedOps::Copy4DIJ2D(FeatureMap, Layers[l + 1].FeatureMap, i, j);
						ManagedMatrix::Rotate180(rot180, FeatureMap);

						ManagedOps::Copy4D3D(Delta, Layers[l + 1].Delta, j);
						ManagedConvolution::Full(Delta, rot180, ztemp);
						ManagedMatrix::Add(z, ztemp);

						ManagedOps::Free(FeatureMap);
						ManagedOps::Free(rot180);
						ManagedOps::Free(ztemp);
					}

					ManagedOps::Copy3D4D(Layers[l].Delta, z, i);

					ManagedOps::Free(Delta);
					ManagedOps::Free(z);
				}
			}
		}

		// calc gradients
		for (auto l = 1; l < n; l++)
		{
			if (Layers[l].Type == LayerTypes::Convolution)
			{
				Layers[l].DeltaFeatureMap.Resize(Layers[l].FeatureMap, false);
				Layers[l].DeltaBias.Resize(Layers[l].OutputMaps, false);

				for (auto j = 0; j < Layers[l].Activation.i; j++)
				{
					// d[j]
					auto dtemp = ManagedArray(Layers[l].Activation.x, Layers[l].Activation.y, FeatureVectorZ, false);
					ManagedOps::Copy4D3D(dtemp, Layers[l].Delta, j);

					for (auto i = 0; i < Layers[l - 1].Activation.i; i++)
					{
						ManagedArray ftemp, FeatureMapDelta;

						// a[i]
						auto atemp = ManagedArray(Layers[l - 1].Activation.x, Layers[l - 1].Activation.y, FeatureVectorZ, false);
						ManagedOps::Copy4D3D(atemp, Layers[l - 1].Activation, i);

						ManagedMatrix::FlipAll(ftemp, atemp);
						ManagedConvolution::Valid(ftemp, dtemp, FeatureMapDelta);
						ManagedMatrix::Multiply(FeatureMapDelta, 1.0 / (double)FeatureVectorZ);

						ManagedOps::Copy2D4DIJ(Layers[l].DeltaFeatureMap, FeatureMapDelta, i, j);

						ManagedOps::Free(FeatureMapDelta);
						ManagedOps::Free(ftemp);
						ManagedOps::Free(atemp);
					}

					Layers[l].DeltaBias(j) = ManagedMatrix::Sum(dtemp) / (double)FeatureVectorZ;

					ManagedOps::Free(dtemp);
				}
			}
		}

		auto FeatureVectorTransposed = ManagedMatrix::Transpose(FeatureVector);

		ManagedMatrix::Multiply(WeightsDelta, OutputDelta, FeatureVectorTransposed);
		ManagedMatrix::Multiply(WeightsDelta, 1.0 / (double)FeatureVectorZ);
		ManagedMatrix::Mean(BiasDelta, OutputDelta, 0);

		ManagedOps::Free(WeightsTransposed);
		ManagedOps::Free(FeatureVectorTransposed);
		ManagedOps::Free(FeatureVector);
		ManagedOps::Free(OutputDelta);
		ManagedOps::Free(OutputError);
	}

	void ApplyGradients(ConvolutionalNeuralNetworkOptions opts)
	{
		for (auto l = 1; l < (int)Layers.size(); l++)
		{
			if (Layers[l].Type == LayerTypes::Convolution)
			{
				ManagedMatrix::Add(Layers[l].FeatureMap, Layers[l].DeltaFeatureMap, -opts.Alpha);
				ManagedMatrix::Add(Layers[l].Bias, Layers[l].DeltaBias, -opts.Alpha);

				ManagedOps::Free(Layers[l].DeltaBias);
				ManagedOps::Free(Layers[l].DeltaFeatureMap);
			}

			ManagedOps::Free(Layers[l].Delta);
		}

		ManagedMatrix::Add(Weights, WeightsDelta, -opts.Alpha);
		ManagedMatrix::Add(Bias, BiasDelta, -opts.Alpha);

		ManagedOps::Free(WeightsDelta);
		ManagedOps::Free(BiasDelta);
	}

	// Classify data using trained network parameters and count classification errors
	int Classify(ManagedArray& test_input, ManagedArray& test_output, int classes, int items, int batchsize, ManagedArray& classification, bool pool = false)
	{
		auto errors = 0;

		auto tempx = ManagedArray(test_input.x, test_input.y, batchsize);
		auto tempy = ManagedArray(batchsize, classes);
		auto tempclass = ManagedArray(1, batchsize);

		ManagedOps::Free(classification);

		classification = ManagedArray(1, items);

		for (auto i = 0; i < items; i += batchsize)
		{
			// generate batch
			ManagedOps::Copy3D(tempx, test_input, 0, 0, i);
			ManagedOps::Copy2D(tempy, test_output, i, 0);

			// classify
			FeedForward(tempx, pool);

			// count classifcation errors
			errors += Test(tempy, tempclass);

			// save classification
			ManagedOps::Copy2DOffset(classification, tempclass, i, 0);
		}

		ManagedOps::Free(tempx);
		ManagedOps::Free(tempy);
		ManagedOps::Free(tempclass);

		return errors;
	}

	// Count classification errors
	int Test(ManagedArray& correct, ManagedArray& classifcation)
	{
		auto errors = 0;

		for (auto x = 0; x < Output.x; x++)
		{
			auto max = 0.0;
			auto cmax = 0.0;
			auto index = 0;
			auto cindex = 0;

			for (auto y = 0; y < Output.y; y++)
			{
				auto val = Output(x, y);

				if (val > max)
				{
					max = val;
					index = y;
				}
			}

			// Save classification
			classifcation(x) = index;

			for (auto cy = 0; cy < Output.y; cy++)
			{
				auto val = correct(x, cy);

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

	void Train(ManagedArray& input, ManagedArray& output, ConvolutionalNeuralNetworkOptions opts)
	{
		auto index_list = ManagedIntList(opts.Items);

		for (auto epoch = 0; epoch < opts.Epochs; epoch++)
		{
			ManagedArray temp_input;
			ManagedArray temp_output;

			if (opts.Shuffle)
			{
				auto seed = static_cast<unsigned int>(time(0));

				srand(seed);

				// TODO: Implement Shuffle
				temp_input.Resize(input.x, input.y, opts.BatchSize, false);
				temp_output.Resize(opts.BatchSize, output.y, false);
			}
			else
			{
				temp_input.Resize(input.x, input.y, opts.BatchSize, false);
				temp_output.Resize(opts.BatchSize, output.y, false);
			}

			auto start = Profiler::now();

			auto batchnum = 0;

			auto rLVal = 0.0;

			rL.clear();

			for (auto i = 0; i < opts.Items; i += opts.BatchSize)
			{
				if (opts.Shuffle)
				{
					// TODO: Implement Shuffle
					ManagedOps::Copy3D(temp_input, input, 0, 0, i);
					ManagedOps::Copy2D(temp_output, output, i, 0);
				}
				else
				{
					ManagedOps::Copy3D(temp_input, input, 0, 0, i);
					ManagedOps::Copy2D(temp_output, output, i, 0);
				}

				FeedForward(temp_input, opts.Pool);
				BackPropagation(temp_output);
				ApplyGradients(opts);

				if (rL.size() == 0)
				{
					rL.push_back(L);
				}

				rLVal = 0.99 * rL[rL.size() - 1] + 0.01 * L;

				rL.push_back(rLVal);

				batchnum++;
			}

			std::cerr << "epoch " << epoch + 1 << "/" << opts.Epochs << " elapsed time is " << Profiler::Elapsed(start) << " ms - Error: " << rLVal << std::endl;

			ManagedOps::Free(temp_input);
			ManagedOps::Free(temp_output);
		}

		ManagedOps::Free(index_list);
	}

	void Free()
	{
		for (auto i = 0; i < (int)Layers.size(); i++)
		{
			ManagedOps::Free(Layers[i].Activation);

			if (Layers[i].Type == LayerTypes::Convolution)
			{
				ManagedOps::Free(Layers[i].FeatureMap);
				ManagedOps::Free(Layers[i].Bias);
			}
		}

		ManagedOps::Free(Weights);
		ManagedOps::Free(Output);
		ManagedOps::Free(Bias);

		rL.clear();
		Layers.clear();
	}
};
#endif
