#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstdio>

#include "Convn.hpp"
#include "MemOps.hpp"

class Matrix
{
public:

	// ------------------------------------------------------------------------------------
	// Print Matrix Elements
	// ------------------------------------------------------------------------------------
	static void PrintList(int* input, int sizex, bool vert = false)
	{
		for (auto x = 0; x < sizex; x++)
		{
			if (!vert)
			{
				if (x > 0)
					printf(" ");
			}
			else
			{
				printf("%d: ", x);
			}

			printf("%d", input[x]);

			if (vert)
				printf("\n");
		}

		if (!vert)
			printf("\n");
	}

	static void Print2D(double* input, int sizex, int sizey)
	{
		for (auto y = 0; y < sizey; y++)
		{
			printf("%d: ", y);

			for (auto x = 0; x < sizex; x++)
			{
				if (x > 0)
				{
					printf(" ");
				}

				printf("%1.2g", MemOps::Val2D(input, sizex, sizey, x, y));
			}

			printf("\n");
		}
	}

	static void Print3D(double* input, int sizex, int sizey, int sizez)
	{
		for (auto z = 0; z < sizez; z++)
		{
			printf("[, , %d]\n", z);

			for (auto y = 0; y < sizey; y++)
			{
				printf("%d: ", y);

				for (auto x = 0; x < sizex; x++)
				{
					if (x > 0)
					{
						printf(" ");
					}

					printf("%1.2g", MemOps::Val3D(input, sizex, sizey, sizez, x, y, z));
				}

				printf("\n");
			}
		}
	}

	static void Print4D(double* input, int sizex, int sizey, int sizez, int i)
	{
		for (auto z = 0; z < sizez; z++)
		{
			printf("[, , %d]\n", z);

			for (auto y = 0; y < sizey; y++)
			{
				printf("%d: ", y);

				for (auto x = 0; x < sizex; x++)
				{
					if (x > 0)
					{
						printf(" ");
					}

					double val = *(input + ((i * sizez + z) * sizey + y) * sizex + x);

					printf("%1.2g", val);
				}

				printf("\n");
			}
		}
	}

	static void Print4D(double* input, int sizex, int sizey, int sizej, int sizei, int i, int j)
	{
		auto temp = MemOps::New(sizex * sizey);

		MemOps::Copy4DIJ2D(temp, input, sizex, sizey, sizej, sizei, i, j);

		Print2D(temp, sizex, sizey);

		MemOps::Free(temp);
	}

	// ------------------------------------------------------------------------------------
	// Matrix Operations
	// ------------------------------------------------------------------------------------

	// 2D Matrix transposition
	static void Transpose(double*& dst, double*& src, int srcx, int srcy)
	{
		for (auto y = 0; y < srcy; y++)
		{
			for (auto x = 0; x < srcx; x++)
			{
				auto val = MemOps::Val2D(src, srcx, srcy, x, y);
				MemOps::Set2D(dst, srcy, srcx, y, x, val);
			}
		}
	}

	// 2D Matrix multiplication
	static void Multiply(double*& result, double*& A, double*& B, int Ax, int Ay, int Bx, int By)
	{
		if (Ax == By)
		{
			for (auto y = 0; y < Ay; y++)
			{
				for (auto x = 0; x < Bx; x++)
				{
					MemOps::Set2D(result, Bx, Ay, x, y, 0.0);

					for (auto k = 0; k < Ax; k++)
					{
						MemOps::Inc2D(result, Bx, Ay, x, y, MemOps::Val2D(A, Ax, Ay, k, y) * MemOps::Val2D(B, Bx, By, x, k));
					}
				}
			}
		}
	}

	// Element by element multiplication
	static void Multiply(double*& A, double*& B, int length)
	{
		for (auto x = 0; x < length; x++)
		{
			A[x] *= B[x];
		}
	}

	// Matrix Addition
	static void Add(double*& A, double*& B, int length, double Scale = 1.0)
	{
		for (auto x = 0; x < length; x++)
		{
			A[x] += Scale * B[x];
		}
	}

	// Matrix * Constant Multiplication
	static void Multiply(double*& A, int length, double B)
	{
		for (auto x = 0; x < length; x++)
		{
			A[x] *= B;
		}
	}

	// Matrix + Constant Addition
	static void Add(double*& A, double B, int length)
	{
		for (auto x = 0; x < length; x++)
		{
			A[x] += B;
		}
	}

	// Matrix Summation
	static double Sum(double*& A, int length)
	{
		auto sum = 0.0;

		for (auto x = 0; x < length; x++)
		{
			sum += A[x];
		}

		return sum;
	}

	// get sum of squares of each element
	static double SquareSum(double*& A, int length)
	{
		auto sum = 0.0;

		for (auto x = 0; x < length; x++)
		{
			sum += A[x] * A[x];
		}

		return sum;
	}

	// Matrix mean of 2D Array along a dimension
	static void Mean(double*& dst, double*& src, int Ax, int Ay, int dim)
	{
		if (dim == 1)
		{
			MemOps::Free(dst);

			dst = MemOps::New(Ax);

			for (auto x = 0; x < Ax; x++)
			{
				auto sum = 0.0;

				for (auto y = 0; y < Ay; y++)
				{
					sum += MemOps::Val2D(src, Ax, Ay, x, y);
				}

				dst[x] = sum / Ay;
			}
		}
		else
		{
			MemOps::Free(dst);

			dst = MemOps::New(Ay);

			for (auto y = 0; y < Ay; y++)
			{
				auto sum = 0.0;

				for (auto x = 0; x < Ax; x++)
				{
					sum += MemOps::Val2D(src, Ax, Ay, x, y);
				}

				dst[y] = sum / Ax;
			}
		}
	}

	// sigmoid function
	static double Sigmoid(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}

	// Apply sigmoid function to matrix
	static double* Sigm(double*& A, int length)
	{
		auto result = MemOps::New(length);

		for (auto x = 0; x < length; x++)
		{
			result[x] = Sigmoid(A[x]);
		}

		return result;
	}

	// Apply delta sigmoid function to matrix
	static double* DSigm(double*& A, int length)
	{
		auto result = MemOps::New(length);

		for (auto x = 0; x < length; x++)
		{
			auto sigmoid = Sigmoid(A[x]);
			result[x] = sigmoid * (1.0 - sigmoid);
		}

		return result;
	}

	// Combine two arrays column-wise
	static double* CBind(double*& A, int Ax, int Ay, double*& B, int Bx, int By)
	{
		if (Ay == By)
		{
			auto resultx = Ax + Bx;
			auto resulty = Ay;

			auto result = MemOps::New(resultx * resulty);

			MemOps::Copy2DOffset(result, A, resultx, resulty, Ax, Ay, 0, 0);
			MemOps::Copy2DOffset(result, B, resultx, resulty, Bx, By, Ax, 0);

			return result;
		}
		else
		{
			return NULL;
		}
	}

	// Flip 2D Matrix along a dimension
	static void Flip(double*& dst, double*& src, int srcx, int srcy, int FlipDim)
	{
		MemOps::Free(dst);

		dst = MemOps::New(srcx * srcy);

		Flip(dst, src, srcx, srcy, 1, FlipDim);
	}

	// Flip 3D Matrix along a dimension
	static void Flip(double*& dst, double*& src, int srcx, int srcy, int srcz, int FlipDim)
	{
		MemOps::Free(dst);

		dst = MemOps::New(srcx * srcy * srcz);

		for (auto z = 0; z < srcz; z++)
		{
			for (auto y = 0; y < srcy; y++)
			{
				for (auto x = 0; x < srcx; x++)
				{
					switch (FlipDim)
					{
					case 0:
						MemOps::Set3D(dst, srcx, srcy, srcz, x, y, z, MemOps::Val3D(src, srcx, srcy, srcz, srcx - x - 1, y, z));
						break;
					case 1:
						MemOps::Set3D(dst, srcx, srcy, srcz, x, y, z, MemOps::Val3D(src, srcx, srcy, srcz, x, srcy - y - 1, z));
						break;
					case 2:
						MemOps::Set3D(dst, srcx, srcy, srcz, x, y, z, MemOps::Val3D(src, srcx, srcy, srcz, x, y, srcz - z - 1));
						break;
					}
				}
			}
		}
	}

	// Flip 3D Matrix along a dimension
	static void FlipAll(double*& dst, double*& src, int srcx, int srcy, int srcz)
	{
		MemOps::Free(dst);

		dst = MemOps::New(srcx * srcy * srcz);

		auto tmp = MemOps::New(srcx * srcy * srcz);

		MemOps::Copy3D(tmp, src, srcx, srcy, srcz, srcx, srcy, srcz, 0, 0, 0);

		for (auto FlipDim = 0; FlipDim < 3; FlipDim++)
		{
			Flip(dst, tmp, srcx, srcy, srcz, FlipDim);

			MemOps::Copy3D(tmp, dst, srcx, srcy, srcz, srcx, srcy, srcz, 0, 0, 0);
		}

		MemOps::Free(tmp);
	}

	// Rotate a 2D matrix
	static void Rotate180(double*& dst, double*& src, int srcx, int srcy)
	{
		MemOps::Free(dst);

		dst = MemOps::New(srcx * srcy);

		auto tmp = MemOps::New(srcx * srcy);

		MemOps::Copy2D(tmp, src, srcx, srcy, srcx, srcy, 0, 0);

		for (auto FlipDim = 0; FlipDim < 2; FlipDim++)
		{
			Flip(dst, tmp, srcx, srcy, FlipDim);

			MemOps::Copy2D(tmp, dst, srcx, srcy, srcx, srcy, 0, 0);
		}

		MemOps::Free(tmp);
	}

	// Expand a matrix A[x][y] by [ex][ey]
	static void Expand(double*& A, int Ax, int Ay, int expandx, int expandy, double*& output)
	{
		auto outputx = Ax * expandx;
		auto outputy = Ay * expandy;

		for (auto y = 0; y < Ay; y++)
		{
			for (auto x = 0; x < Ax; x++)
			{
				for (auto SZy = 0; SZy < expandy; SZy++)
				{
					for (auto SZx = 0; SZx < expandx; SZx++)
					{
						MemOps::Set2D(output, outputx, outputy, x * expandx + SZx, y * expandy + SZy, MemOps::Val2D(A, Ax, Ay, x, y));
					}
				}
			}
		}
	}

	// Create a 2D Diagonal/Identity matrix of size [dim][dim]
	static double* Diag(int dim)
	{
		if (dim > 0)
		{
			auto result = MemOps::New(dim * dim);

			for (auto y = 0; y < dim; y++)
			{
				for (auto x = 0; x < dim; x++)
				{
					if (x == y)
					{
						MemOps::Set2D(result, dim, dim, x, y, 1.0);
					}
					else
					{
						MemOps::Set2D(result, dim, dim, x, y, 0.0);
					}
				}
			}

			return result;
		}
		else
		{
			return NULL;
		}
	}
};
#endif
