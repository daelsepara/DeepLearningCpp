#ifndef MANAGED_DNN_HPP
#define MANAGED_DNN_HPP

#include <vector>

#include "HiddenLayer.hpp"
#include "ManagedArray.hpp"
#include "ManagedFile.hpp"
#include "ManagedMatrix.hpp"
#include "ManagedOps.hpp"
#include "NeuralNetworkOptions.hpp"
#include "Random.hpp"

class ManagedDNN
{
public:

	std::vector<HiddenLayer> Layers;
	std::vector<ManagedArray> Weights;
	std::vector<ManagedArray> Deltas;
	
	// intermediate results
	std::vector<ManagedArray> X;
	std::vector<ManagedArray> Z;
	
	// internal use
	std::vector<ManagedArray> Activations;
	std::vector<ManagedArray> D;

	ManagedArray Y;
	ManagedArray Y_true;

	std::vector<double> Min;
	std::vector<double> Max;

	// Error
	double Cost;
	double L2;

	int Iterations;

	// Forward Propagation
	void Forward(ManagedArray& input)
	{
		// add bias column to input layer
		auto InputBias = ManagedArray(1, input.y);
		ManagedOps::Set(InputBias, 1.0);

		// Compute input activations
		int last = (int)Weights.size() - 1;

		for (int layer = 0; layer < (int)Weights.size(); layer++)
		{
			auto tW = ManagedMatrix::Transpose(Weights[layer]);
			ManagedOps::Free(X[layer]);
			ManagedOps::Free(Z[layer]);

			X[layer] = layer == 0 ? ManagedMatrix::CBind(InputBias, input) : ManagedMatrix::CBind(InputBias, Activations[layer - 1]);
			Z[layer] = ManagedMatrix::Multiply(X[layer], tW);

			ManagedOps::Free(tW);

			if (layer != last)
			{
				ManagedOps::Free(Activations[layer]);

				Activations[layer] = ManagedMatrix::Sigm(Z[layer]);
			}
			else
			{
				ManagedOps::Free(Y);

				Y = ManagedMatrix::Sigm(Z[layer]);
			}
		}

		// Cleanup
		for (int layer = 0; layer < Activations.size(); layer++)
		{
			ManagedOps::Free(Activations[layer]);
		}

		ManagedOps::Free(InputBias);
	}

	// Backward propagation
	void BackPropagation(ManagedArray& input)
	{
		int last = (int)Weights.size() - 1;

		ManagedOps::Free(D[0]);

		D[0] = ManagedMatrix::Diff(Y, Y_true);

		int current = 1;

		for (int layer = last - 1; layer >= 0; layer--)
		{
			int prev = current - 1;

			auto W = ManagedArray(Weights[layer + 1].x - 1, Weights[layer + 1].y, false);
			auto DZ = ManagedMatrix::DSigm(Z[layer]);

			ManagedOps::Free(D[current]);

			D[current] = (ManagedArray(W.x, D[prev].y, false));

			ManagedOps::Copy2D(W, Weights[layer + 1], 1, 0);
			ManagedMatrix::Multiply(D[current], D[prev], W);
			ManagedMatrix::Product(D[current], DZ);

			ManagedOps::Free(W);
			ManagedOps::Free(DZ);

			current++;
		}

		for (int layer = 0; layer < Weights.size(); layer++)
		{
			auto tD = ManagedMatrix::Transpose(D[Weights.size() - layer - 1]);

			ManagedOps::Free(Deltas[layer]);

			Deltas[layer] = (ManagedArray(Weights[layer].x, Weights[layer].y, false));

			ManagedMatrix::Multiply(Deltas[layer], tD, X[layer]);
			ManagedMatrix::Multiply(Deltas[layer], 1.0 / input.y);

			ManagedOps::Free(tD);
		}

		Cost = 0.0;
		L2 = 0.0;

		for (int i = 0; i < Y_true.Length(); i++)
		{
			L2 += 0.5 * (D[0](i) * D[0](i));
			Cost += (-Y_true(i) * log(Y(i)) - (1 - Y_true(i)) * log(1 - Y(i)));
		}

		Cost /= input.y;
		L2 /= input.y;

		// Cleanup
		for (int layer = 0; layer < Weights.size(); layer++)
		{
			ManagedOps::Free(D[layer]);
			ManagedOps::Free(X[layer]);
			ManagedOps::Free(Z[layer]);
		}
	}

	void ClearDeltas()
	{
		for (int layer = 0; layer < Weights.size(); layer++)
		{
			// cleanup of arrays allocated in BackPropagation
			ManagedOps::Free(Deltas[layer]);
		}
	}

	void ApplyGradients(NeuralNetworkOptions opts)
	{
		for (int layer = 0; layer < Weights.size(); layer++)
		{
			ManagedMatrix::Add(Weights[layer], Deltas[layer], -opts.Alpha);
		}
	}

	void Rand(ManagedArray& rnd, Random random)
	{
		for (int x = 0; x < rnd.Length(); x++)
		{
			rnd(x) = (random.NextDouble() - 0.5) * 2.0;
		}
	}

	ManagedArray Labels(ManagedArray& output, NeuralNetworkOptions opts)
	{
		auto result = ManagedArray(opts.Categories, opts.Items);
		auto eye_matrix = ManagedMatrix::Diag(opts.Categories);

		for (int y = 0; y < opts.Items; y++)
		{
			if (opts.Categories > 1)
			{
				for (int x = 0; x < opts.Categories; x++)
				{
					result(x, y) = eye_matrix(x, (int)output(y) - 1);
				}
			}
			else
			{
				result(y) = (int)output(y);
			}
		}

		ManagedOps::Free(eye_matrix);

		return result;
	}

	ManagedArray Predict(ManagedArray& test)
	{
		Forward(test);

		auto prediction = ManagedArray(test.y);

		for (int y = 0; y < test.y; y++)
		{
			if (Y.x > 1)
			{
				double maxval = std::numeric_limits<double>::lowest();

				for (int x = 0; x < Y.x; x++)
				{
					double val = Y(x, y);

					if (val > maxval)
					{
						maxval = val;
					}
				}

				prediction(y) = maxval;
			}
			else
			{
				prediction(y) = Y(y);
			}
		}

		// cleanup of arrays allocated in Forward propagation
		ManagedOps::Free(Y);

		// Cleanup
		for (int layer = 0; layer < Weights.size(); layer++)
		{
			ManagedOps::Free(X[layer]);
			ManagedOps::Free(Z[layer]);
		}

		return prediction;
	}

	ManagedIntList Classify(ManagedArray& test, double threshold = 0.5)
	{
		Forward(test);

		auto classification = ManagedIntList(test.y);

		for (int y = 0; y < test.y; y++)
		{
			if (Y.x > 1)
			{
				double maxval = std::numeric_limits<double>::lowest();
				int maxind = 0;

				for (int x = 0; x < Y.x; x++)
				{
					double val = Y(x, y);

					if (val > maxval)
					{
						maxval = val;
						maxind = x;
					}
				}

				classification(y) = maxind + 1;
			}
			else
			{
				classification(y) = Y(y) > threshold ? 1 : 0;
			}
		}

		// cleanup of arrays allocated in Forward propagation
		ManagedOps::Free(Y);

		for (int layer = 0; layer < Weights.size(); layer++)
		{
			ManagedOps::Free(X[layer]);
			ManagedOps::Free(Z[layer]);
		}

		return classification;
	}

	void SetupLabels(ManagedArray& output, NeuralNetworkOptions opts)
	{
		Y_true = Labels(output, opts);
	}

	void SetupHiddenLayers(int inputs, int categories, std::vector<int> LayerNodes)
	{
		if (LayerNodes.size() > 0)
		{
			if (Layers.size() > 0)
			{
				Layers.clear();
			}

			Layers.push_back(HiddenLayer(inputs, LayerNodes[0]));

			for (int layer = 1; layer < LayerNodes.size(); layer++)
			{
				Layers.push_back(HiddenLayer(LayerNodes[layer - 1], LayerNodes[layer]));
			}

			Layers.push_back(HiddenLayer(LayerNodes[LayerNodes.size() - 1], categories));
		}
	}

	void Setup(ManagedArray& output, NeuralNetworkOptions opts, bool Reset = true)
	{
		if (Reset)
		{
			if (Activations.size() > 0)
			{
				for (int layer = 0; layer < Activations.size(); layer++)
				{
					ManagedOps::Free(Activations[layer]);
				}
			}

			if (D.size() > 0)
			{
				for (int layer = 0; layer < D.size(); layer++)
				{
					ManagedOps::Free(D[layer]);
				}
			}

			if (Deltas.size() > 0)
			{
				for (int layer = 0; layer < Deltas.size(); layer++)
				{
					ManagedOps::Free(Deltas[layer]);
				}
			}

			if (X.size() > 0)
			{
				for (int layer = 0; layer < X.size(); layer++)
				{
					ManagedOps::Free(X[layer]);
				}
			}

			if (Z.size() > 0)
			{
				for (int layer = 0; layer < Z.size(); layer++)
				{
					ManagedOps::Free(Z[layer]);
				}
			}

			if (Weights.size() > 0)
			{
				for (int layer = 0; layer < Weights.size(); layer++)
				{
					ManagedOps::Free(Weights[layer]);
				}
			}

			if (Layers.size() > 0)
			{
				Weights.resize(Layers.size());

				for (int layer = 0; layer < Layers.size(); layer++)
				{
					Weights[layer] = ManagedArray(Layers[layer].Inputs + 1, Layers[layer].Outputs);
				}
			}
			else
			{
				// Regenerate Layers list
				
				Weights.resize(opts.HiddenLayers + 1);

				Weights[0] = ManagedArray(opts.Inputs + 1, opts.Nodes);
				
				Layers.push_back(HiddenLayer(opts.Inputs, opts.Nodes));

				for (int layer = 1; layer < opts.HiddenLayers; layer++)
				{
					Weights[layer] = ManagedArray(opts.Nodes + 1, opts.Nodes);
					
					Layers.push_back(HiddenLayer(opts.Nodes, opts.Nodes));
				}

				Weights[opts.HiddenLayers] = ManagedArray(opts.Nodes + 1, opts.Categories);
				
				Layers.push_back(HiddenLayer(opts.Nodes, opts.Categories));
			}
		}

		Activations.resize(opts.HiddenLayers);
		Deltas.resize(opts.HiddenLayers + 1);
		X.resize(opts.HiddenLayers + 1);
		D.resize(opts.HiddenLayers + 1);
		Z.resize(opts.HiddenLayers + 1);

		SetupLabels(output, opts);

		auto random = Random();
		random.UniformDistribution();

		if (Reset && Weights.size() > 0)
		{
			for (int layer = 0; layer < opts.HiddenLayers + 1; layer++)
			{
				Rand(Weights[layer], random);
			}
		}

		Cost = 1.0;
		L2 = 1.0;

		Iterations = 0;
	}

	ManagedArray Normalize(ManagedArray& input)
	{
		Min.clear();
		Max.clear();

		auto result = ManagedArray(input.x, input.y, false);

		for (int i = 0; i < input.x; i++)
		{
			Max.push_back(std::numeric_limits<double>::min());
			Min.push_back(std::numeric_limits<double>::max());
		}

		for (int y = 0; y < input.y; y++)
		{
			for (int x = 0; x < input.x; x++)
			{
				auto val = input(x, y);

				Max[x] = std::max(Max[x], val);
				Min[x] = std::min(Min[x], val);
			}
		}

		for (int y = 0; y < input.y; y++)
		{
			for (int x = 0; x < input.x; x++)
			{
				auto val = input(x, y);

				auto denum = Max[x] - Min[x];

				result(x, y) = (val - Min[x]) / denum;
			}
		}

		return result;
	}

	ManagedArray ApplyNormalization(ManagedArray& input)
	{
		auto result = ManagedArray(input.x, input.y, false);

		if (Min.size() > 0 && Max.size() > 0)
		{
			for (int y = 0; y < input.y; y++)
			{
				for (int x = 0; x < input.x; x++)
				{
					auto val = input(x, y);

					auto denum = Max[x] - Min[x];

					result(x, y) = (val - Min[x]) / denum;
				}
			}
		}
		else
		{
			for (int y = 0; y < input.y; y++)
			{
				for (int x = 0; x < input.x; x++)
				{
					result(x, y) = input(x, y);
				}
			}
		}

		return result;
	}

	bool Step(ManagedArray& input, NeuralNetworkOptions opts)
	{
		Forward(input);
		BackPropagation(input);

		bool optimized = (std::isnan(opts.UseL2 ? L2 : Cost) || (opts.UseL2 ? L2 : Cost) < opts.Tolerance);

		// Apply gradients only if the error is still high
		if (!optimized)
		{
			ApplyGradients(opts);
		}

		ClearDeltas();

		Iterations = Iterations + 1;

		return (optimized || Iterations >= opts.Epochs);
	}

	void Train(ManagedArray& input, ManagedArray& output, NeuralNetworkOptions opts)
	{
		Setup(output, opts);

		while (!Step(input, opts)) {}
	}

	void Free()
	{
		ManagedOps::Free(Y);
		ManagedOps::Free(Y_true);

		if (Weights.size() > 0)
		{
			for (int layer = 0; layer < Weights.size(); layer++)
			{
				ManagedOps::Free(Weights[layer]);
			}
		}

		if (Layers.size() > 0)
			Layers.clear();

		if (Activations.size() > 0)
			Activations.clear();

		if (X.size() > 0)
			X.clear();

		if (Z.size() > 0)
			Z.clear();

		if (Deltas.size() > 0)
			Deltas.clear();

		if (D.size() > 0)
			D.clear();
	}
};
#endif
