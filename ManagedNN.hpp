#ifndef MANAGED_NN_HPP
#define MANAGED_NN_HPP

#include <vector>
#include <limits>

#include "FuncOutput.hpp"
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

private:

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

	void Rand(ManagedArray& rnd, Random random)
	{
		for (auto x = 0; x < rnd.Length(); x++)
		{
			rnd(x) = (random.NextDouble() - 0.5) * 2.0;
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

public:

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

	ManagedIntList Classify(ManagedArray& test, double threshold = 0.5)
	{
		Forward(test);

		auto classification = ManagedIntList(test.y);

		for (auto y = 0; y < test.y; y++)
		{
			if (Y.x > 1)
			{
				auto maxval = 0.0;
				auto maxind = 0;

				for (auto x = 0; x < Y.x; x++)
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

	ManagedArray Predict(ManagedArray& test)
	{
		Forward(test);

		auto prediction = ManagedArray(test.y);

		for (auto y = 0; y < test.y; y++)
		{
			if (Y.x > 1)
			{
				auto maxval = 0.0;

				for (auto x = 0; x < Y.x; x++)
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
	
	void SetupNetwork(ManagedArray& output, NeuralNetworkOptions opts)
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

private:

	bool StepNetwork(ManagedArray& input, NeuralNetworkOptions opts)
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

public:

	void Train(ManagedArray& input, ManagedArray& output, NeuralNetworkOptions opts)
	{
		SetupNetwork(output, opts);
		
		while (!StepNetwork(input, opts)) {}
	}

	// Minimize a continuous differentialble multivariate function. Starting point
	// is given by "X" (D by 1), and the function named in the string "f", must
	// return a function value and a vector of partial derivatives. The Polack-
	// Ribiere flavour of conjugate gradients is used to compute search directions,
	// and a line search using quadratic and cubic polynomial approximations and the
	// Wolfe-Powell stopping criteria is used together with the slope ratio method
	// for guessing initial step sizes. Additionally a bunch of checks are made to
	// make sure that exploration is taking place and that extrapolation will not
	// be unboundedly large. The "length" gives the length of the run: if it is
	// positive, it gives the maximum number of line searches, if negative its
	// absolute gives the maximum allowed number of function evaluations. You can
	// (optionally) give "length" a second component, which will indicate the
	// reduction in function value to be expected in the first line-search (defaults
	// to 1.0). The function returns when either its length is up, or if no further
	// progress can be made (ie, we are at a minimum, or so close that due to
	// numerical problems, we cannot get any closer). If the function terminates
	// within a few iterations, it could be an indication that the function value
	// and derivatives are not consistent (ie, there may be a bug in the
	// implementation of your "f" function). The function returns the found
	// solution "X", a vector of function values "fX" indicating the progress made
	// and "i" the number of iterations (line searches or function evaluations,
	// depending on the sign of "length") used.
	//
	// Usage: [X, fX, i] = fmincg(f, X, options)
	//
	// See also: checkgrad 
	//
	// Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
	//
	//
	// (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
	// 
	// Permission is granted for anyone to copy, use, or modify these
	// programs and accompanying documents for purposes of research or
	// education, provided this copyright notice is retained, and note is
	// made of any changes that have been made.
	// 
	// These programs and documents are distributed without any warranty,
	// express or implied.  As the programs were written for research
	// purposes only, they have not been tested to the degree that would be
	// advisable in any important application.  All use of these programs is
	// entirely at the user's own risk.
	//
	// Original C# implementation by Peter Sergio Larsen to work with Accord.NET framework
	// see: https://github.com/accord-net/framework/blob/master/Sources/Extras/Accord.Math.Noncommercial/NonlinearConjugateGradient.cs
	//
	// Changes by [sdsepara, 2018]: 
	//
	// 1) Function to minimize must return a result of type FuncOutput (see above)
	// 2) success and ls_failed changed to type bool, and M to type int. 
	// 3) modified to work with NeuralNetworkClassifier
	// 4) each call to StepOptimizer executes just one cycle of optimization
	// 5) implemented Multiply, Add, Copy helper functions
	//
	// C++ implementation by [sdsepara, 2019]
	//

private:

	// Generate vector from array for use in optimizer
	std::vector<double> ReshapeWeights(ManagedArray& A, ManagedArray& B)
	{
		std::vector<double> XX;

		int index = 0;

		for (int x = 0; x < A.x; x++)
		{
			for (int y = 0; y < A.y; y++)
			{
				XX.push_back(A(x, y));

				index++;
			}
		}

		for (int x = 0; x < B.x; x++)
		{
			for (int y = 0; y < B.y; y++)
			{
				XX.push_back(B(x, y));

				index++;
			}
		}

		return XX;
	}

	// Transform vector back into Network Weights
	void TransferWeights(std::vector<double> XX)
	{
		int index = 0;

		for (int x = 0; x < Wkj.x; x++)
		{
			for (int y = 0; y < Wkj.y; y++)
			{
				if (index < XX.size())
					Wkj(x, y) = XX[index];

				index++;
			}
		}

		for (int x = 0; x < Wji.x; x++)
		{
			for (int y = 0; y < Wji.y; y++)
			{
				if (index < XX.size())
					Wji(x, y) = XX[index];

				index++;
			}
		}
	}

	FuncOutput OptimizerCost(ManagedArray& input, std::vector<double> XX)
	{
		TransferWeights(XX);

		Forward(input);
		BackPropagation(input);

		auto DD = ReshapeWeights(DeltaWkj, DeltaWji);

		ClearDeltas();

		return FuncOutput(Cost, DD);
	}

	double Multiply(std::vector<double> a, std::vector<double> b)
	{
		if (a.size() == b.size())
		{
			auto dot = 0.0;

			for (auto i = 0; i < a.size(); i++)
				dot += a[i] * b[i];

			return dot;
		}

		return 0.0;
	}

	void Add(std::vector<double> & dst, std::vector<double> src, double scale = 1)
	{
		if (dst.size() == src.size())
		{
			for (auto i = 0; i < dst.size(); i++)
				dst[i] += scale * src[i];
		}
	}

	void Copy(std::vector<double> & dst, std::vector<double> src, double scale = 1)
	{
		if (dst.size() == src.size())
		{
			for (auto i = 0; i < dst.size(); i++)
				dst[i] = scale * src[i];
		}
	}

	// RHO and SIG are the constants in the Wolfe-Powell conditions
	const double RHO = 0.01;
	const double SIG = 0.5;

	// don't reevaluate within 0.1 of the limit of the current bracket
	const double INT = 0.1;

	// extrapolate maximum 3 times the current bracket
	const double EXT = 3.0;

	// max 20 function evaluations per line search
	const int MAX = 20;

	// maximum allowed slope ratio
	const double RATIO = 100.0;

	// reduction parameter
	const double Red = 1.0;

	std::vector<double> s;
	std::vector<double> df1;

	int MaxIterations;
	int Evaluations;

	int length;
	int M;
	int iteration;
	bool ls_failed;

	double f1;

	std::vector<double> X0;
	std::vector<double> DF0;

	double d1;
	double z1;

	void SetupOptimizer(ManagedArray& input)
	{
		auto XX = ReshapeWeights(Wkj, Wji);

		s.resize(XX.size());

		Evaluations = 0;
		Iterations = 0;

		length = MaxIterations;
		M = 0;
		iteration = 0; // zero the run length counter
		ls_failed = false; // no previous line search has failed

		// get function value and gradient
		auto eval = OptimizerCost(input, XX);
		f1 = eval.Error;
		df1 = eval.Gradient;

		Evaluations++;

		// count epochs?!
		if (length < 0)
			iteration++;

		// search direction is steepest
		Copy(s, df1, -1.0);

		// this is the slope
		d1 = -Multiply(s, s);

		// initial step is red / (|s|+1)
		z1 = Red / (1.0 - d1);

		X0.resize(XX.size());
		DF0.resize(XX.size());
	}

	bool StepOptimizer(ManagedArray& input)
	{
		auto XX = ReshapeWeights(Wkj, Wji);

		// from R/Matlab smallest non-zero normalized floating point number
		auto realmin = 2.225074e-308;

		// count iterations?!
		if (length > 0)
			iteration++;

		Iterations = iteration;

		// make a copy of current values
		Copy(X0, XX);
		Copy(DF0, df1);

		auto F0 = f1;

		// begin line search
		Add(XX, s, z1);

		// evaluate cost - and gradient function with new params
		auto eval = OptimizerCost(input, XX);

		auto f2 = eval.Error;
		auto df2 = eval.Gradient;

		Evaluations++;

		// count epochs?!
		if (length < 0)
			iteration++;

		// initialize point 3 equal to point 1
		auto d2 = Multiply(df2, s);

		auto f3 = f1;
		auto d3 = d1;
		auto z3 = -z1;

		if (length > 0)
		{
			M = MAX;
		}
		else
		{
			M = std::min(MAX, -length - iteration);
		}

		// initialize quantities
		auto success = false;
		auto limit = -1.0;

		while (true)
		{
			while (((f2 > f1 + z1 * RHO * d1) || (d2 > -SIG * d1)) && (M > 0))
			{
				// tighten bracket
				limit = z1;

				auto A = 0.0;
				auto B = 0.0;
				auto z2 = 0.0;

				if (f2 > f1)
				{
					// quadratic fit 
					z2 = z3 - ((0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3));
				}
				else
				{
					// cubic fit
					A = (6.0 * (f2 - f3)) / (z3 + (3.0 * (d2 + d3)));
					B = (3.0 * (f3 - f2) - (z3 * ((d3 + 2.0) * d2)));

					// numerical error possible - ok!
					z2 = std::sqrt(((B * B) - (A * d2 * z3)) - B) / A;
				}

				if (std::isnan(z2) || std::isinf(z2))
				{
					// if we had a numerical problem then bisect
					z2 = z3 / 2.0;
				}

				// don't accept too close to limit
				z2 = std::max(std::min(z2, INT * z3), (1.0 - INT) * z3);

				// update the step
				z1 = z1 + z2;

				Add(XX, s, z2);

				eval = OptimizerCost(input, XX);
				f2 = eval.Error;
				df2 = eval.Gradient;
				Evaluations++;

				M = M - 1;

				// count epochs?!
				if (length < 0)
					iteration++;

				d2 = Multiply(df2, s);

				// z3 is now relative to the location of z2
				z3 = z3 - z2;
			}

			if (f2 > (f1 + z1 * RHO * d1) || d2 > (-SIG * d1))
			{
				// this is a failure
				break;
			}

			if (d2 > (SIG * d1))
			{
				// success
				success = true;

				break;
			}

			if (M == 0)
			{
				// failure
				break;
			}

			// make cubic extrapolation
			auto A1 = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3);
			auto B1 = 3.0 * (f3 - f2) - z3 * (d3 + 2.0 * d2);

			// num error possible - ok!
			auto z21 = -d2 * z3 * z3 / (B1 + std::sqrt(B1 * B1 - A1 * d2 * z3 * z3));

			if (z21 < 0.0)
			{
				z21 = z21 * -1.0;
			}

			// num prob or wrong sign?
			if (std::isnan(z21) || std::isinf(z21) || z21 < 0)
			{
				// if we have no upper limit
				if (limit < -0.5)
				{
					// then extrapolate the maximum amount
					z21 = z1 * (EXT - 1.0);
				}
				else
				{
					// otherwise bisect
					z21 = (limit - z1) / 2.0;
				}
			}
			else if (limit > -0.5 && (z21 + z1 > limit))
			{
				// extrapolation beyond limit?

				// set to extrapolation limit
				z21 = (limit - z1) / 2.0;
			}
			else if (limit < -0.5 && (z21 + z1 > z1 * EXT))
			{
				z21 = z1 * (EXT - 1.0);
			}
			else if (z21 < -z3 * INT)
			{
				// too close to limit?
				z21 = -z3 * INT;
			}
			else if ((limit > -0.5) && (z21 < (limit - z1) * (1 - INT)))
			{
				z21 = (limit - z1) * (1.0 - INT);
			}

			// set point 3 equal to point 2
			f3 = f2;
			d3 = d2;
			z3 = -z21;
			z1 = z1 + z21;

			// update current estimates
			Add(XX, s, z21);

			// evaluate functions
			eval = OptimizerCost(input, XX);
			df2 = eval.Gradient;
			f2 = eval.Error;

			M = M - 1;

			// count epochs?!
			iteration = iteration + (length < 0 ? 1 : 0);

			d2 = Multiply(df2, s);

			// end of line search
		}

		// if line searched succeeded 
		if (success)
		{
			f1 = f2;

			// Polack-Ribiere direction
			auto part1 = Multiply(df2, df2);
			auto part2 = Multiply(df1, df2);
			auto part3 = Multiply(df1, df1);

			Copy(s, s, (part1 - part2) / part3);
			Add(s, df2, -1.0);

			// swap derivatives
			auto tmp = df1;
			df1 = df2;
			df2 = tmp;

			// get slope
			d2 = Multiply(df1, s);

			// new slope must be negative 
			if (d2 > 0.0)
			{
				// use steepest direction
				Copy(s, df1, -1.0);

				d2 = -Multiply(s, s);
			}

			// slope ratio but max RATIO
			z1 = z1 * std::min(RATIO, (d1 / (d2 - realmin)));

			d1 = d2;

			// this line search did not fail
			ls_failed = false;
		}
		else
		{
			// restore point from before failed line search
			f1 = F0;

			Copy(XX, X0);
			Copy(df1, DF0);

			// line search twice in a row
			if (ls_failed || iteration > std::abs(length))
			{
				// or we ran out of time, so we give up
				return true;
			}

			// swap derivatives
			auto tmp = df1;
			df1 = df2;
			df2 = tmp;

			// try steepest
			Copy(s, df1, -1.0);

			d1 = -Multiply(s, s);

			z1 = 1.0 / (1.0 - d1);

			// this line search failed
			ls_failed = true;
		}

		return !(iteration < std::abs(length));
	}

	bool Optimized(ManagedArray& input, NeuralNetworkOptions opts)
	{
		auto SearchComplete = StepOptimizer(input);

		Cost = f1;

		return (SearchComplete || std::isnan(Cost) || Iterations >= opts.Epochs || (Cost) < opts.Tolerance);
	}

public:

	void Optimize(ManagedArray& input, ManagedArray& output, NeuralNetworkOptions opts)
	{
		SetupNetwork(output, opts);

		MaxIterations = opts.Epochs;

		SetupOptimizer(input);

		while (!Optimized(input, opts)) {}
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
