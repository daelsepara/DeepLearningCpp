#ifndef MANAGED_MATRIX_HPP
#define MANAGED_MATRIX_HPP

#include <iostream>
#include <iomanip>

#include "ManagedArray.hpp"
#include "ManagedOps.hpp"

class ManagedMatrix
{
public:

	// ------------------------------------------------------------------------------------
	// Print Matrix Elements
	// ------------------------------------------------------------------------------------
	static void PrintList(ManagedIntList& input, bool vert = false)
	{
		for (auto x = 0; x < input.x; x++)
		{
			if (!vert)
			{
				if (x > 0)
					std::cerr << " ";
			}
			else
			{
				std::cerr << x << ": ";
			}

			std::cerr << input(x);

			if (vert)
				std::cerr << std::endl;
		}

		if (!vert)
			std::cerr << std::endl;
	}

	static void Print2D(ManagedArray& input)
	{
		for (auto y = 0; y < input.y; y++)
		{
			std::cerr << y << ": ";

			for (auto x = 0; x < input.x; x++)
			{
				if (x > 0)
				{
					std::cerr << " ";
				}

				std::cerr << std::setprecision(2) << std::scientific << input(x, y);
			}

			std::cerr << std::endl;
		}
	}

	static void Print3D(ManagedArray& input)
	{
		for (auto z = 0; z < input.z; z++)
		{
			std::cerr << "[, , " << z << "]" << std::endl;

			for (auto y = 0; y < input.y; y++)
			{
				std::cerr << y << ": ";

				for (auto x = 0; x < input.x; x++)
				{
					if (x > 0)
					{
						std::cerr << " ";
					}

					std::cerr << std::setprecision(2) << std::scientific << input(x, y, z);
				}

				std::cerr << std::endl;
			}
		}
	}

	static void Print4D(ManagedArray& input, int i)
	{
		for (auto z = 0; z < input.z; z++)
		{
			std::cerr << "[, , " << z << "]" << std::endl;

			for (auto y = 0; y < input.y; y++)
			{
				std::cerr << y << ": ";

				for (auto x = 0; x < input.x; x++)
				{
					if (x > 0)
					{
						std::cerr << " ";
					}

					std::cerr << std::setprecision(2) << std::scientific << input(((i * input.z + z) * input.y + y) * input.x + x);
				}

				std::cerr << std::endl;
			}
		}
	}

	static void Print4DIJ(ManagedArray& input, int i, int j)
	{
		auto size2D = input.x * input.y;
		auto srcoffset = (i * input.j + j) * size2D;

		for (auto y = 0; y < input.y; y++)
		{
			std::cerr << y << ": ";

			for (auto x = 0; x < input.x; x++)
			{
				if (x > 0)
				{
					std::cerr << " ";
				}

				std::cerr << std::setprecision(2) << std::scientific << input(srcoffset + y * input.x + x);
			}

			std::cerr << std::endl;
		}
	}

	// ------------------------------------------------------------------------------------
	// Matrix Operations
	// ------------------------------------------------------------------------------------

	// 2D Matrix transposition
	static void Transpose(ManagedArray& dst, ManagedArray& src)
	{
		dst.Resize(src.y, src.x, false);

		for (auto y = 0; y < src.y; y++)
		{
			for (auto x = 0; x < src.x; x++)
			{
				dst(y, x) = src(x, y);
			}
		}
	}

	static ManagedArray Transpose(ManagedArray& src)
	{
		auto dst = ManagedArray(src.y, src.x, false);

		Transpose(dst, src);

		return dst;
	}

	#if !defined (FAST_MATRIX_MULTIPLY)

	// 2D Matrix multiplication - Naive Version
	static void Multiply(ManagedArray& result, ManagedArray& A, ManagedArray& B)
	{
		if (A.x == B.y)
		{
			// Naive version
			result.Resize(B.x, A.y, false);

			for (auto y = 0; y < A.y; y++)
			{
				for (auto x = 0; x < B.x; x++)
				{
					result(x, y) = 0.0;

					for (auto k = 0; k < A.x; k++)
					{
						result(x, y) += A(k, y) * B(x, k);
					}
				}
			}
		}
	}

	#else

	// 2D Matrix multiplication
	// slightly faster (due to memory access pattern) but still naive
	// see: https://tavianator.com/a-quick-trick-for-faster-naive-matrix-multiplication/
	static void Multiply(ManagedArray& result, ManagedArray& A, ManagedArray& B)
	{
		if (A.x == B.y)
		{
			auto dest = 0;
			auto lhs = 0;
			auto rhs = 0;
			auto mid = A.x;
			auto cols = B.x;
			auto rows = A.y;

			result.Resize(cols, rows, true);

			for (auto y = 0; y < rows; y++)
			{
				rhs = 0;

				for (auto x = 0; x < mid; x++)
				{
					auto lhsx = lhs + x;

					for (auto k = 0; k < cols; k++)
					{
						result(dest + k) += A(lhsx) * B(rhs + k);
					}

					rhs += cols;
				}

				dest += cols;
				lhs += mid;
			}
		}
	}
	#endif

	// 2D Matrix multiplication
	static ManagedArray Multiply(ManagedArray& A, ManagedArray& B)
	{
		ManagedArray result;

		Multiply(result, A, B);

		return result;
	}

	static ManagedArray Pow(ManagedArray& A, double power)
	{
		auto result = ManagedArray(A.x, A.y, A.z, A.i, A.j, false);

		for (auto x = 0; x < A.Length(); x++)
		{
			result(x) = std::pow(A(x), power);
		}

		return result;
	}

	static ManagedArray Pow(double A, ManagedArray& powers)
	{
		auto result = ManagedArray(powers.x, powers.y, powers.z, powers.i, powers.j, false);

		for (auto i = 0; i < powers.Length(); i++)
		{
			result(i) = std::pow(A, powers(i));
		}

		return result;
	}

	// Matrix * Constant Multiplication
	static void Multiply(ManagedArray& A, double B)
	{
		for (auto x = 0; x < A.Length(); x++)
		{
			A(x) *= B;
		}
	}

	// Element by element multiplication
	static void Product(ManagedArray& result, ManagedArray& A, ManagedArray& B)
	{
		for (auto x = 0; x < A.Length(); x++)
		{
			result(x) = A(x) * B(x);
		}
	}

	// Element by element multiplication
	static void Product(ManagedArray& A, ManagedArray& B)
	{
		Product(A, A, B);
	}

	// Element by element multiplication
	static ManagedArray BSXMUL(ManagedArray& A, ManagedArray& B)
	{
		auto result = ManagedArray(A.x, A.y, A.z, A.i, A.j, false);

		for (auto x = 0; x < A.Length(); x++)
		{
			result(x) = A(x) * B(x);
		}

		return result;
	}

	// Matrix Addition
	static void Add(ManagedArray& A, ManagedArray& B, double Scale = 1.0)
	{
		for (auto x = 0; x < A.Length(); x++)
		{
			A(x) += Scale * B(x);
		}
	}

	// Matrix + Constant Addition
	static void Add(ManagedArray& A, double B)
	{
		for (auto x = 0; x < A.Length(); x++)
		{
			A(x) += B;
		}
	}

	// Matrix Summation
	static double Sum(ManagedArray& A)
	{
		auto sum = 0.0;

		for (auto x = 0; x < A.Length(); x++)
		{
			sum += A(x);
		}

		return sum;
	}

	// get sum of squares of each element
	static double SquareSum(ManagedArray& A)
	{
		auto sum = 0.0;

		for (auto x = 0; x < A.Length(); x++)
		{
			sum += A(x) * A(x);
		}

		return sum;
	}

	// Matrix mean of 2D Array along a dimension
	static void Mean(ManagedArray& dst, ManagedArray& src, int dim)
	{
		if (dim == 1)
		{
			dst.Resize(src.x, 1, false);

			for (auto x = 0; x < src.x; x++)
			{
				auto sum = 0.0;

				for (auto y = 0; y < src.y; y++)
				{
					sum += src(x, y);
				}

				dst(x) = sum / (double)src.y;
			}
		}
		else
		{
			dst.Resize(1, src.y, false);

			for (auto y = 0; y < src.y; y++)
			{
				auto sum = 0.0;

				for (auto x = 0; x < src.x; x++)
				{
					sum += src(x, y);
				}

				dst(y) = sum / (double)src.x;
			}
		}
	}

	// sigmoid function
	static double Sigmoid(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	// Get element per element difference between arrays
	static ManagedArray Diff(ManagedArray& A, ManagedArray& B)
	{
		auto result = ManagedArray(A.x, A.y, A.z, A.i, A.j, false);

		for (auto x = 0; x < A.Length(); x++)
		{
			result(x) = A(x) - B(x);
		}

		return result;
	}

	// Apply sigmoid function to matrix
	static ManagedArray Sigm(ManagedArray& A)
	{
		auto result = ManagedArray(A.x, A.y, A.z, A.i, A.j, false);

		for (auto x = 0; x < A.Length(); x++)
		{
			result(x) = Sigmoid(A(x));
		}

		return result;
	}

	// Apply delta sigmoid function to matrix
	static ManagedArray DSigm(ManagedArray& A)
	{
		auto result = ManagedArray(A.x, A.y, A.z, A.i, A.j, false);

		for (auto x = 0; x < A.Length(); x++)
		{
			auto sigmoid = Sigmoid(A(x));
			result(x) = sigmoid * (1 - sigmoid);
		}

		return result;
	}

	// Combine two arrays column-wise
	static ManagedArray CBind(ManagedArray& A, ManagedArray& B)
	{
		if (A.y == B.y)
		{
			auto resultx = A.x + B.x;
			auto resulty = A.y;

			auto result = ManagedArray(resultx, resulty, false);

			ManagedOps::Copy2DOffset(result, A, 0, 0);
			ManagedOps::Copy2DOffset(result, B, A.x, 0);

			return result;
		}
		else
		{
			return NULL;
		}
	}

	// Flip Matrix along a dimension
	static void Flip(ManagedArray& dst, ManagedArray& src, int FlipDim)
	{
		dst.Resize(src.x, src.y, src.z, false);

		for (auto z = 0; z < src.z; z++)
		{
			for (auto y = 0; y < src.y; y++)
			{
				for (auto x = 0; x < src.x; x++)
				{
					switch (FlipDim)
					{

					case 0:
						dst(x, y, z) = src(src.x - x - 1, y, z);
						break;
					case 1:
						dst(x, y, z) = src(x, src.y - y - 1, z);
						break;
					case 2:
						dst(x, y, z) = src(x, y, src.z - z - 1);
						break;
					}
				}
			}
		}
	}

	// Flip 3D Matrix along a dimension
	static void FlipAll(ManagedArray& dst, ManagedArray& src)
	{
		dst.Resize(src.x, src.y, src.z, false);

		auto tmp = ManagedArray(src.x, src.y, src.z, false);

		ManagedOps::Copy3D(tmp, src, 0, 0, 0);

		for (auto FlipDim = 0; FlipDim < 3; FlipDim++)
		{
			Flip(dst, tmp, FlipDim);

			ManagedOps::Copy3D(tmp, dst, 0, 0, 0);
		}

		ManagedOps::Free(tmp);
	}

	// Rotate a 2D matrix
	static void Rotate180(ManagedArray& dst, ManagedArray& src)
	{
		dst.Resize(src.x, src.y, false);

		auto tmp = ManagedArray(src.x, src.y, false);

		ManagedOps::Copy2D(tmp, src, 0, 0);

		for (auto FlipDim = 0; FlipDim < 2; FlipDim++)
		{
			Flip(dst, tmp, FlipDim);

			ManagedOps::Copy2D(tmp, dst, 0, 0);
		}

		ManagedOps::Free(tmp);
	}

	// Expand a matrix A[x][y] by [ex][ey]
	static void Expand(ManagedArray& A, int expandx, int expandy, ManagedArray& output)
	{
		auto outputx = A.x * expandx;
		auto outputy = A.y * expandy;

		output.Resize(outputx, outputy, false);

		for (auto y = 0; y < A.y; y++)
		{
			for (auto x = 0; x < A.x; x++)
			{
				for (auto SZy = 0; SZy < expandy; SZy++)
				{
					for (auto SZx = 0; SZx < expandx; SZx++)
					{
						output(x * expandx + SZx, y * expandy + SZy) = A(x, y);
					}
				}
			}
		}
	}

	// Expand a matrix A[x][y] by [ex][ey]
	static ManagedArray Expand(ManagedArray& A, int expandx, int expandy)
	{
		ManagedArray output;

		Expand(A, expandx, expandy, output);

		return output;
	}

	// Transforms x into a column vector
	static void Vector(ManagedArray& x)
	{
		auto temp = Transpose(x);

		x.Reshape(1, x.Length());

		for (auto i = 0; i < x.Length(); i++)
		{
			x(i) = temp(i);
		}

		ManagedOps::Free(temp);
	}

	static ManagedArray RowSums(ManagedArray& A)
	{
		auto result = ManagedArray(1, A.y);

		for (auto i = 0; i < A.y; i++)
		{
			result(i) = 0.0;

			for (auto j = 0; j < A.x; j++)
			{
				result(i) += A(j, i);
			}
		}

		return result;
	}

	static ManagedArray ColSums(ManagedArray& A)
	{
		auto result = ManagedArray(A.x, 1);

		for (auto j = 0; j < A.x; j++)
		{
			result(j) = 0.0;

			for (auto i = 0; i < A.y; i++)
			{
				result(j) += A(j, i);
			}
		}

		return result;
	}

	// Create a 2D Diagonal/Identity matrix of size [dim][dim]
	static ManagedArray Diag(int dim)
	{
		if (dim > 0)
		{
			auto result = ManagedArray(dim, dim);

			for (auto y = 0; y < dim; y++)
			{
				for (auto x = 0; x < dim; x++)
				{
					result(x, y) = (x == y) ? 1.0 : 0.0;
				}
			}

			return result;
		}
		else
		{
			return NULL;
		}
	}

	static void Sqrt(ManagedArray& x)
	{
		for (auto i = 0; i < x.Length(); i++)
			x(i) = std::sqrt(x(i));
	}
};
#endif
