#ifndef CONVN_HPP
#define CONVN_HPP

#include <cmath>

#include "ConvolutionShape.hpp"
#include "MemOps.hpp"

class Convn
{
public:

	// Convolution between 2D input and filter
	static void Apply(double*& input, double*& filter, double*& result, int inputx, int inputy, int filterx, int filtery, int& outputx, int& outputy, ConvolutionShape shape)
	{
		auto outputz = 1;

		if (shape == ConvolutionShape::Full)
		{
			Full(input, filter, result, inputx, inputy, 1, filterx, filtery, 1, outputx, outputy, outputz);
		}
		else if (shape == ConvolutionShape::Same)
		{
			Same(input, filter, result, inputx, inputy, 1, filterx, filtery, 1, outputx, outputy, outputz);
		}
		else if (shape == ConvolutionShape::Valid)
		{
			Valid(input, filter, result, inputx, inputy, 1, filterx, filtery, 1, outputx, outputy, outputz);
		}
	}

	// Convolution between 3D array with a 2D filter
	static void Apply(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int& outputx, int& outputy, int& outputz, ConvolutionShape shape)
	{
		Apply(input, filter, result, inputx, inputy, inputz, filterx, filtery, 1, outputx, outputy, outputz, shape);
	}

	// Convolution between 3D array with a 3D filter
	static void Apply(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int filterz, int& outputx, int& outputy, int& outputz, ConvolutionShape shape)
	{
		if (shape == ConvolutionShape::Valid)
		{
			Valid(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, outputx, outputy, outputz);
		}
		else if (shape == ConvolutionShape::Full)
		{
			Full(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, outputx, outputy, outputz);
		}
		else if (shape == ConvolutionShape::Same)
		{
			Same(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, outputx, outputy, outputz);
		}
	}

	static void Full(double*& input, double*& filter, double*& result, int inputx, int inputy, int filterx, int filtery)
	{
		auto outputx = 0, outputy = 0, outputz = 0;

		Full(input, filter, result, inputx, inputy, 1, filterx, filtery, 1, outputx, outputy, outputz);
	}

	static void Full(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int filterz)
	{
		auto outputx = 0, outputy = 0, outputz = 0;

		Full(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, outputx, outputy, outputz);
	}

	static void Full(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int filterz, int& outputx, int& outputy, int& outputz)
	{
		auto cx = inputx + filterx - 1;
		auto cy = inputy + filtery - 1;
		auto cz = inputz + filterz - 1;

		Convolve(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, 0, 0, 0, cx, cy, cz, outputx, outputy, outputz);
	}

	static void Same(double*& input, double*& filter, double*& result, int inputx, int inputy, int filterx, int filtery)
	{
		auto outputx = 0, outputy = 0, outputz = 0;

		Same(input, filter, result, inputx, inputy, 1, filterx, filtery, 1, outputx, outputy, outputz);
	}

	static void Same(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int filterz)
	{
		auto outputx = 0, outputy = 0, outputz = 0;

		Same(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, outputx, outputy, outputz);
	}

	static void Same(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int filterz, int& outputx, int& outputy, int& outputz)
	{
		auto cx = inputx + filterx - 1;
		auto cy = inputy + filtery - 1;
		auto cz = inputz + filterz - 1;

		auto dx = (double)(filterx - 1) / 2;
		auto dy = (double)(filtery - 1) / 2;
		auto dz = (double)(filterz - 1) / 2;

		auto minx = (int)ceil(dx);
		auto miny = (int)ceil(dy);
		auto minz = (int)ceil(dz);

		auto maxx = (int)ceil((double)cx - dx - 1);
		auto maxy = (int)ceil((double)cy - dy - 1);
		auto maxz = (int)ceil((double)cz - dz - 1);

		auto limx = maxx - minx + 1;
		auto limy = maxy - miny + 1;
		auto limz = maxz - minz + 1;

		Convolve(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, minx, miny, minz, limx, limy, limz, outputx, outputy, outputz);
	}

	static void Valid(double*& input, double*& filter, double*& result, int inputx, int inputy, int filterx, int filtery)
	{
		auto outputx = 0, outputy = 0, outputz = 0;

		Valid(input, filter, result, inputx, inputy, 1, filterx, filtery, 1, outputx, outputy, outputz);
	}

	static void Valid(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int filterz)
	{
		auto outputx = 0, outputy = 0, outputz = 0;

		Valid(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, outputx, outputy, outputz);
	}

	static void Valid(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int filterz, int& outputx, int& outputy, int& outputz)
	{
		auto minx = filterx - 1;
		auto miny = filtery - 1;
		auto minz = filterz - 1;

		auto limx = inputx - filterx + 1;
		auto limy = inputy - filtery + 1;
		auto limz = inputz - filterz + 1;

		Convolve(input, filter, result, inputx, inputy, inputz, filterx, filtery, filterz, minx, miny, minz, limx, limy, limz, outputx, outputy, outputz);
	}

	static void Convolve(double*& input, double*& filter, double*& result, int inputx, int inputy, int inputz, int filterx, int filtery, int filterz, int minx, int miny, int minz, int limx, int limy, int limz, int& outputx, int& outputy, int& outputz)
	{
		MemOps::Free(result);

		result = MemOps::New(limx * limy * limz);

		outputx = 0;
		outputy = 0;
		outputz = 0;

		if (inputx >= filterx && inputy >= filtery && inputz >= filterz)
		{
			for (auto ck = minz; ck < minz + limz; ck++)
			{
				for (auto cj = miny; cj < miny + limy; cj++)
				{
					for (auto ci = minx; ci < minx + limx; ci++)
					{
						MemOps::Set3D(result, limx, limy, limz, ci - minx, cj - miny, ck - minz, 0.0);

						for (auto kz = 0; kz < inputz; kz++)
						{
							auto boundz = ck - kz;

							if (boundz >= 0 && boundz < filterz && kz < inputz && kz >= 0)
							{
								for (auto ky = 0; ky < inputy; ky++)
								{
									auto boundy = cj - ky;

									if (boundy >= 0 && boundy < filtery && ky < inputy && ky >= 0)
									{
										for (auto kx = 0; kx < inputx; kx++)
										{
											auto boundx = ci - kx;

											if (boundx >= 0 && boundx < filterx && kx < inputx && kx >= 0)
											{
												auto fval = MemOps::Val3D(filter, filterx, filtery, filterz, boundx, boundy, boundz);
												auto ival = MemOps::Val3D(input, inputx, inputy, inputz, kx, ky, kz);

												MemOps::Inc3D(result, limx, limy, limz, ci - minx, cj - miny, ck - minz, ival * fval);
											}
										}
									}
								}
							}
						}
					}
				}
			}

			outputx = limx;
			outputy = limy;
			outputz = limz;
		}
	}
};

#endif
