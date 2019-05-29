#ifndef MANAGED_CONVOLUTION_HPP
#define MANAGED_CONVOLUTION_HPP

#include <cmath>

#include "ConvolutionShape.hpp"
#include "ManagedArray.hpp"

class ManagedConvolution
{
public:

	static void Full(ManagedArray& input, ManagedArray& filter, ManagedArray& result)
	{
		auto cx = input.x + filter.x - 1;
		auto cy = input.y + filter.y - 1;
		auto cz = input.z + filter.z - 1;

		Convolve(input, filter, result, 0, 0, 0, cx, cy, cz);
	}

	static void Same(ManagedArray& input, ManagedArray& filter, ManagedArray& result)
	{
		auto cx = input.x + filter.x - 1;
		auto cy = input.y + filter.y - 1;
		auto cz = input.z + filter.z - 1;

		auto dx = (double)(filter.x - 1) / 2;
		auto dy = (double)(filter.y - 1) / 2;
		auto dz = (double)(filter.z - 1) / 2;

		auto minx = (int)std::ceil(dx);
		auto miny = (int)std::ceil(dy);
		auto minz = (int)std::ceil(dz);

		auto maxx = (int)std::ceil((double)cx - dx - 1);
		auto maxy = (int)std::ceil((double)cy - dy - 1);
		auto maxz = (int)std::ceil((double)cz - dz - 1);

		auto limx = maxx - minx + 1;
		auto limy = maxy - miny + 1;
		auto limz = maxz - minz + 1;

		Convolve(input, filter, result, minx, miny, minz, limx, limy, limz);
	}

	static void Valid(ManagedArray& input, ManagedArray& filter, ManagedArray& result)
	{
		auto minx = filter.x - 1;
		auto miny = filter.y - 1;
		auto minz = filter.z - 1;

		auto limx = input.x - filter.x + 1;
		auto limy = input.y - filter.y + 1;
		auto limz = input.z - filter.z + 1;

		Convolve(input, filter, result, minx, miny, minz, limx, limy, limz);
	}

	static void Convolve(ManagedArray& input, ManagedArray& filter, ManagedArray& result, int minx, int miny, int minz, int limx, int limy, int limz)
	{
		result.Resize(limx, limy, limz, false);

		if (input.x >= filter.x && input.y >= filter.y && input.z >= filter.z)
		{
			for (auto ck = minz; ck < minz + limz; ck++)
			{
				for (auto cj = miny; cj < miny + limy; cj++)
				{
					for (auto ci = minx; ci < minx + limx; ci++)
					{
						result(ci - minx, cj - miny, ck - minz) = 0.0;

						for (auto kz = 0; kz < input.z; kz++)
						{
							auto boundz = ck - kz;

							if (boundz >= 0 && boundz < filter.z && kz < input.z && kz >= 0)
							{
								for (auto ky = 0; ky < input.y; ky++)
								{
									auto boundy = cj - ky;

									if (boundy >= 0 && boundy < filter.y && ky < input.y && ky >= 0)
									{
										for (auto kx = 0; kx < input.x; kx++)
										{
											auto boundx = ci - kx;

											if (boundx >= 0 && boundx < filter.x && kx < input.x && kx >= 0)
											{
												auto fval = filter(boundx, boundy, boundz);
												auto ival = input(kx, ky, kz);

												result(ci - minx, cj - miny, ck - minz) += ival * fval;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
};

#endif
