#ifndef MANAGED_NN_HPP
#define MANAGED_NN_HPP

#include <vector>

#include "ManagedArray.hpp"
#include "ManagedFile.hpp"
#include "ManagedMatrix.hpp"
#include "ManagedOps.hpp"
#include "NeuralNetworkOptions.hpp"
#include "Random.hpp"

class ManagedNN
{
public:

	ManagedArray Wji = NULL;
	ManagedArray Wkj = NULL;

	// intermediate results
	ManagedArray Y = NULL;
	ManagedArray Y_true = NULL;
	ManagedArray Zji = NULL;
	ManagedArray Aji = NULL;
	ManagedArray DeltaWji = NULL;
	ManagedArray DeltaWkj = NULL;

	std::vector<double> Min;
	std::vector<double> Max;

	double Cost = 0.0;
	double L2 = 0.0;

	int Iterations = 0;

	// Forward Propagation
	void Forward(ManagedArray& input)
	{
		auto Bias = ManagedArray(1, input.y);
		ManagedOps::Set(Bias, 1.0);

		// add bias column to input layer
		auto x = ManagedMatrix::CBind(Bias, input);

		// compute hidden layer activation
		ManagedOps::Free(Zji);
		auto tWji = ManagedMatrix::Transpose(Wji);
		Zji = ManagedMatrix::Multiply(x, tWji);

		auto Z = ManagedMatrix::Sigm(Zji);

		// add bias column to hidden layer output
		ManagedOps::Free(Aji);
		Aji = ManagedMatrix::CBind(Bias, Z);

		// compute output layer
		auto tWkj = ManagedMatrix::Transpose(Wkj);
		auto AjiWkj = ManagedMatrix::Multiply(Aji, tWkj);

		ManagedOps::Free(Y);
		Y = ManagedMatrix::Sigm(AjiWkj);

		// cleanup
		ManagedOps::Free(AjiWkj);
		ManagedOps::Free(Bias);
		ManagedOps::Free(tWkj);
		ManagedOps::Free(tWji);
		ManagedOps::Free(x);
		ManagedOps::Free(Z);
	}

	// Backward propagation
	void BackPropagation(ManagedArray& input)
	{
		// add bias column to input layer
		auto Bias = ManagedArray(1, input.y);
		ManagedOps::Set(Bias, 1.0);
		auto x = ManagedMatrix::CBind(Bias, input);

		// compute intermediate delta values per layer
		auto Difference = ManagedMatrix::Diff(Y, Y_true);

		auto sWkj = ManagedArray(Wkj.x - 1, Wkj.y);
		ManagedOps::Copy2D(sWkj, Wkj, 1, 0);
		
		auto Dkj = ManagedMatrix::Multiply(Difference, sWkj);
		auto DZji = ManagedMatrix::DSigm(Zji);
		ManagedMatrix::Product(Dkj, DZji);

		auto tDkj = ManagedMatrix::Transpose(Dkj);
		auto tDifference = ManagedMatrix::Transpose(Difference);

		ManagedOps::Free(DeltaWji);
		ManagedOps::Free(DeltaWkj);

		DeltaWji = ManagedMatrix::Multiply(tDkj, x);
		DeltaWkj = ManagedMatrix::Multiply(tDifference, Aji);

		ManagedMatrix::Multiply(DeltaWji, 1.0 / input.y);
		ManagedMatrix::Multiply(DeltaWkj, 1.0 / input.y);

		Cost = 0.0;
		L2 = 0.0;
		
		for (auto i = 0; i < Y_true.Length(); i++)
		{
			L2 += 0.5 * (Difference(i) * Difference(i));
			Cost += (-Y_true(i) * log(Y(i)) - (1.0 - Y_true(i)) * log(1.0 - Y(i)));
		}

		Cost /= input.y;
		L2 /= input.y;

		// cleanup
		ManagedOps::Free(Aji);
		ManagedOps::Free(Zji);
		ManagedOps::Free(Bias);
		ManagedOps::Free(x);
		ManagedOps::Free(Dkj);
		ManagedOps::Free(Difference);
		ManagedOps::Free(DZji);
		ManagedOps::Free(sWkj);
		ManagedOps::Free(tDkj);
		ManagedOps::Free(tDifference);
	}
	
	void ClearDeltas()
	{
		ManagedOps::Free(DeltaWji);
		ManagedOps::Free(DeltaWkj);
	}
	
	void ApplyGradients(NeuralNetworkOptions opts)
	{
		ManagedMatrix::Add(Wkj, DeltaWkj, -opts.Alpha);
		ManagedMatrix::Add(Wji, DeltaWji, -opts.Alpha);
	}

	void Rand(ManagedArray& rand, Random random)
	{
		for (auto x = 0; x < rand.Length(); x++)
		{
			rand(x) = (random.NextDouble() - 0.5) * 2.0;
		}
	}

	ManagedArray Labels(ManagedArray& output, NeuralNetworkOptions opts)
	{
		auto result = ManagedArray(opts.Categories, opts.Items);
		auto eye_matrix = ManagedMatrix::Diag(opts.Categories);

		for (auto y = 0; y < opts.Items; y++)
		{
			if (opts.Categories > 1)
			{
				for (auto x = 0; x < opts.Categories; x++)
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

	ManagedIntList Classify(ManagedArray& test, NeuralNetworkOptions opts, double threshold = 0.5)
	{
		Forward(test);

		auto classification = ManagedIntList(test.y);

		for (auto y = 0; y < test.y; y++)
		{
			if (opts.Categories > 1)
			{
				auto maxval = 0.0;
				auto maxind = 0;

				for (auto x = 0; x < opts.Categories; x++)
				{
					auto val = Y(x, y);

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

		ManagedOps::Free(Aji);
		ManagedOps::Free(Y);
		ManagedOps::Free(Zji);

		return classification;
	}

	ManagedArray Predict(ManagedArray& test, NeuralNetworkOptions opts)
	{
		Forward(test);

		auto prediction = ManagedArray(test.y);

		for (auto y = 0; y < test.y; y++)
		{
			if (opts.Categories > 1)
			{
				auto maxval = 0.0;

				for (auto x = 0; x < opts.Categories; x++)
				{
					auto val = Y(x, y);

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

		ManagedOps::Free(Aji);
		ManagedOps::Free(Y);
		ManagedOps::Free(Zji);

		return prediction;
	}
	
	void Setup(ManagedArray& output, NeuralNetworkOptions opts)
	{
		
		Wji = ManagedArray(opts.Inputs + 1, opts.Nodes);
		Wkj = ManagedArray(opts.Nodes + 1, opts.Categories);

		Y_true = Labels(output, opts);

		auto random = Random();
		random.UniformDistribution();

		Rand(Wji, random);
		Rand(Wkj, random);

		Cost = 1.0;
		Iterations = 0;
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
		ManagedOps::Free(Wji);
		ManagedOps::Free(Wkj);
	}
};
#endif
