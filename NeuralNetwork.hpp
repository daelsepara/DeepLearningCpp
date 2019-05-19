#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>

#include "Matrix.hpp"
#include "NeuralNetworkOptions.hpp"
#include "Random.hpp"

class NeuralNetwork
{
public:

	double* Wji = NULL;
	double* Wkj = NULL;

	// intermediate results
	double* Y = NULL;
	double* Y_true = NULL;
	double* Zji = NULL;
	double* Aji = NULL;
	double* DeltaWji = NULL;
	double* DeltaWkj = NULL;

	std::vector<double> Min;
	std::vector<double> Max;

	double Cost = 0.0;
	double L2 = 0.0;
	
	int Iterations = 0;

	// Forward Propagation
	void Forward(double*& input, NeuralNetworkOptions opts)
	{
		auto Bias = MemOps::New(opts.Items);
		MemOps::Set(Bias, opts.Items, 1.0);

		// add bias column to input layer
		auto x = Matrix::CBind(Bias, 1, opts.Items, input, opts.Inputs, opts.Items);

		// compute hidden layer activation
		MemOps::Free(Zji);
		Zji = MemOps::New(opts.Nodes, opts.Items);

		auto tWji = MemOps::New(opts.Nodes, opts.Inputs + 1);
		Matrix::Transpose(tWji, Wji, opts.Inputs + 1, opts.Nodes);
		Matrix::Multiply(Zji, x, tWji, opts.Inputs + 1, opts.Items, opts.Nodes, opts.Inputs + 1);

		auto Z = Matrix::Sigm(Zji, opts.Nodes * opts.Items);

		// add bias column to hidden layer output
		MemOps::Free(Aji);
		Aji = Matrix::CBind(Bias, 1, opts.Items, Z, opts.Nodes, opts.Items);

		// compute output layer
		auto tWkj = MemOps::New(opts.Categories, opts.Nodes + 1);
		Matrix::Transpose(tWkj, Wkj, opts.Nodes + 1, opts.Categories);

		auto AjiWkj = MemOps::New(opts.Categories, opts.Items);
		Matrix::Multiply(AjiWkj, Aji, tWkj, 1 + opts.Nodes, opts.Items, opts.Categories, opts.Nodes + 1);

		MemOps::Free(Y);
		Y = Matrix::Sigm(AjiWkj, opts.Categories * opts.Items);

		// cleanup
		MemOps::Free(AjiWkj);
		MemOps::Free(Bias);
		MemOps::Free(tWkj);
		MemOps::Free(tWji);
		MemOps::Free(x);
		MemOps::Free(Z);
	}

	// Backward propagation
	void BackPropagation(double*& input, NeuralNetworkOptions opts)
	{
		auto Bias = MemOps::New(opts.Items);
		MemOps::Set(Bias, opts.Items, 1.0);

		// add bias column			
		auto x = Matrix::CBind(Bias, 1, opts.Items, input, opts.Inputs, opts.Items);

		// compute intermediate delta values per layer
		auto Difference = MemOps::New(opts.Categories, opts.Items);
		auto Dkj = MemOps::New(opts.Nodes, opts.Items);

		for (auto i = 0; i < opts.Categories * opts.Items; i++)
		{
			Difference[i] = Y[i] - Y_true[i];
		}

		auto sWkj = MemOps::New(opts.Nodes, opts.Categories);
		MemOps::Copy2D(sWkj, Wkj, opts.Nodes, opts.Categories, opts.Nodes + 1, opts.Categories, 1, 0);
		Matrix::Multiply(Dkj, Difference, sWkj, opts.Categories, opts.Items, opts.Nodes, opts.Categories);

		auto DZji = Matrix::DSigm(Zji, opts.Nodes * opts.Items);
		Matrix::Multiply(Dkj, DZji, opts.Nodes * opts.Items);

		MemOps::Free(DeltaWji);
		MemOps::Free(DeltaWkj);

		DeltaWji = MemOps::New(opts.Inputs + 1, opts.Nodes);
		DeltaWkj = MemOps::New(opts.Nodes + 1, opts.Categories);

		auto tDkj = MemOps::New(opts.Nodes, opts.Items);
		auto tDifference = MemOps::New(opts.Categories, opts.Items);
		Matrix::Transpose(tDkj, Dkj, opts.Nodes, opts.Items);
		Matrix::Transpose(tDifference, Difference, opts.Categories, opts.Items);

		Matrix::Multiply(DeltaWji, tDkj, x, opts.Items, opts.Nodes, opts.Inputs + 1, opts.Items);
		Matrix::Multiply(DeltaWkj, tDifference, Aji, opts.Items, opts.Categories, 1 + opts.Nodes, opts.Items);

		Matrix::Multiply(DeltaWji, (opts.Inputs + 1) * opts.Nodes, 1.0 / opts.Items);
		Matrix::Multiply(DeltaWkj, (1 + opts.Nodes) * opts.Categories, 1.0 / opts.Items);

		Cost = 0.0;
		L2 = 0.0;
		
		for (auto i = 0; i < opts.Categories * opts.Items; i++)
		{
			L2 += 0.5 * (Difference[i] * Difference[i]);
			Cost += (-Y_true[i] * log(Y[i]) - (1.0 - Y_true[i]) * log(1.0 - Y[i]));
		}

		Cost /= opts.Items;
		L2 /= opts.Items;

		// cleanup
		MemOps::Free(Aji);
		MemOps::Free(Dkj);
		MemOps::Free(Difference);
		MemOps::Free(DZji);
		MemOps::Free(Bias);
		MemOps::Free(sWkj);
		MemOps::Free(tDkj);
		MemOps::Free(tDifference);
		MemOps::Free(x);
		MemOps::Free(Zji);
	}

	void ApplyGradients(NeuralNetworkOptions opts)
	{
		Matrix::Add(Wkj, DeltaWkj, (opts.Nodes + 1) * opts.Categories, -opts.Alpha);
		Matrix::Add(Wji, DeltaWji, (opts.Inputs + 1) * opts.Nodes, -opts.Alpha);

		MemOps::Free(DeltaWji);
		MemOps::Free(DeltaWkj);
	}

	void Rand(double*& rand, Random random, int length)
	{
		for (auto x = 0; x < length; x++)
		{
			rand[x] = (random.NextDouble() - 0.5) * 2.0;
		}
	}

	double* Labels(double*& output, NeuralNetworkOptions opts)
	{
		auto result = MemOps::New(opts.Categories, opts.Items);
		auto eye_matrix = Matrix::Diag(opts.Categories);

		for (auto y = 0; y < opts.Items; y++)
		{
			if (opts.Categories > 1)
			{
				for (auto x = 0; x < opts.Categories; x++)
				{
					MemOps::Set2D(result, opts.Categories, opts.Items, x, y, MemOps::Val2D(eye_matrix, opts.Categories, opts.Categories, x, (int)output[y] - 1));
				}
			}
			else
			{
				result[y] = output[y];
			}
		}

		MemOps::Free(eye_matrix);

		return result;
	}

	double* Normalize(double*& input, NeuralNetworkOptions opts)
	{
		Min.clear();
		Max.clear();

		auto result = MemOps::New(opts.Inputs, opts.Items, false);

		for (int i = 0; i < opts.Inputs; i++)
		{
			Max.push_back(std::numeric_limits<double>::min());
			Min.push_back(std::numeric_limits<double>::max());
		}

		for (int y = 0; y < opts.Items; y++)
		{
			for (int x = 0; x < opts.Inputs; x++)
			{
				auto val = MemOps::Val2D(input, opts.Inputs, opts.Items, x, y);

				Max[x] = std::max(Max[x], val);
				Min[x] = std::min(Min[x], val);
			}
		}

		for (int y = 0; y < opts.Items; y++)
		{
			for (int x = 0; x < opts.Inputs; x++)
			{
				auto val = MemOps::Val2D(input, opts.Inputs, opts.Items, x, y);

				auto denum = Max[x] - Min[x];

				MemOps::Set2D(result, opts.Inputs, opts.Items, x, y, (val - Min[x]) / denum);
			}
		}

		return result;
	}

	double* ApplyNormalization(double*& input, NeuralNetworkOptions opts)
	{
		auto result = MemOps::New(opts.Inputs, opts.Items, false);

		if (Min.size() > 0 && Max.size() > 0)
		{
			for (int y = 0; y < opts.Items; y++)
			{
				for (int x = 0; x < opts.Inputs; x++)
				{
					auto val = MemOps::Val2D(input, opts.Inputs, opts.Items, x, y);

					auto denum = Max[x] - Min[x];

					MemOps::Set2D(result, opts.Inputs, opts.Items, x, y, (val - Min[x]) / denum);
				}
			}
		}
		else
		{
			for (int y = 0; y < opts.Items; y++)
			{
				for (int x = 0; x < opts.Inputs; x++)
				{
					auto val = MemOps::Val2D(input, opts.Inputs, opts.Items, x, y);

					MemOps::Set2D(result, opts.Inputs, opts.Items, x, y, val);
				}
			}
		}

		return result;
	}

	int* Classify(double*& test, NeuralNetworkOptions test_opts, double threshold = 0.5)
	{
		Forward(test, test_opts);

		auto classification = MemOps::IntList(test_opts.Items);

		for (auto y = 0; y < test_opts.Items; y++)
		{
			if (test_opts.Categories > 1)
			{
				auto maxval = 0.0;
				auto maxind = 0;

				for (auto x = 0; x < test_opts.Categories; x++)
				{
					auto val = MemOps::Val2D(Y, test_opts.Categories, test_opts.Items, x, y);

					if (val > maxval)
					{
						maxval = val;
						maxind = x;
					}
				}

				classification[y] = maxind + 1;
			}
			else
			{
				classification[y] = Y[y] > threshold ? 1 : 0;
			}
		}

		MemOps::Free(Aji);
		MemOps::Free(Y);
		MemOps::Free(Zji);

		return classification;
	}

	void Train(double*& input, double*& output, NeuralNetworkOptions opts)
	{
		Wji = MemOps::New(opts.Inputs + 1, opts.Nodes);
		Wkj = MemOps::New(opts.Nodes + 1, opts.Categories);

		Y_true = Labels(output, opts);

		auto random = Random();
		random.UniformDistribution();

		Rand(Wji, random, (opts.Inputs + 1) * opts.Nodes);
		Rand(Wkj, random, (opts.Nodes + 1) * opts.Categories);

		Cost = 1.0;
		Iterations = 0;

		while (!std::isnan(Cost) && Iterations < opts.Epochs && Cost > opts.Tolerance)
		{
			Forward(input, opts);
			BackPropagation(input, opts);
			ApplyGradients(opts);

			Iterations = Iterations + 1;
		}

		MemOps::Free(Y_true);
	}

	void Free()
	{
		MemOps::Free(Y);
		MemOps::Free(Y_true);
		MemOps::Free(Wji);
		MemOps::Free(Wkj);
	}
};

#endif
