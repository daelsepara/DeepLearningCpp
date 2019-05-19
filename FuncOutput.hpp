#ifndef FUNC_OUTPUT_HPP
#define FUNC_OUTPUT_HPP

#include <vector>

class FuncOutput
{
public:

	double Error;
	std::vector<double> Gradient;

	FuncOutput(double error, std::vector<double> X)
	{
		Gradient = X;
		Error = error;
	}
};
#endif
