#ifndef MEMOPS_HPP
#define MEMOPS_HPP

class MemOps
{
public:

	static void MemCopy(double*& dst, int dstoffset, double*& src, int srcoffset, int count)
	{
		for (auto i = 0; i < count; i++)
			dst[dstoffset + i] = src[srcoffset + i];
	}

	static void Set(double*& dst, int length, double value)
	{
		for (auto i = 0; i < length; i++) dst[i] = value;
	}

	static double* New(int size, bool initialize = true)
	{
		auto temp = new double[size];

		if (initialize)
		{
			for (auto i = 0; i < size; i++)
				temp[i] = 0.0;
		}

		return temp;
	}

	static double* New(int sizex, int sizey, bool initialize = true)
	{
		return New(sizex * sizey, initialize);
	}
	
	static double* New(int sizex, int sizey, int sizez, int sizei, int sizej, bool initialize = true)
	{
		return New(sizex * sizey * sizez * sizei * sizej, initialize);
	}
	
	static int* IntList(int size)
	{
		auto temp = new int[size];

		for (auto i = 0; i < size; i++)
			temp[i] = i;

		return temp;
	}

	static double Val2D(double* input, int sizex, int sizey, int x, int y)
	{
		return input[y * sizex + x];
	}

	static void Set2D(double* input, int sizex, int sizey, int x, int y, double val)
	{
		input[y * sizex + x] = val;
	}

	static void Inc2D(double* input, int sizex, int sizey, int x, int y, double val)
	{
		input[y * sizex + x] += val;
	}

	static double Val3D(double* input, int sizex, int sizey, int sizez, int x, int y, int z)
	{
		return input[(z * sizey + y) * sizex + x];
	}

	static void Set3D(double* input, int sizex, int sizey, int sizez, int x, int y, int z, double val)
	{
		input[(z * sizey + y) * sizex + x] = val;
	}

	static void Inc3D(double* input, int sizex, int sizey, int sizez, int x, int y, int z, double val)
	{
		input[(z * sizey + y) * sizex + x] += val;
	}

	// Copy 2D[minx + x][miny + y]
	static void Copy2D(double*& dst, double*& src, int dstx, int dsty, int srcx, int srcy, int minx, int miny)
	{
		if (miny >= 0 && miny < srcy)
		{
			for (auto y = 0; y < dsty; y++)
			{
				auto srcoffset = (miny + y) * srcx + minx;
				auto dstoffset = y * dstx;

				MemCopy(dst, dstoffset, src, srcoffset, dstx);
			}
		}
	}

	// Copy 2D[index_list][y]
	static void Copy2DX(double*& dst, double*& src, int*& index_list, int dstx, int dsty, int srcx, int srcy, int minx)
	{
		for (auto y = 0; y < dsty; y++)
		{
			auto dstoffset = y * dstx;
			auto srcoffset = y * srcx;

			auto xx = index_list[minx];

			MemCopy(dst, dstoffset, src, srcoffset + xx, dstx);
		}
	}

	// Copy 2D[x][y] to 2D[minx + x][miny + y]
	static void Copy2DOffset(double*& dst, double*& src, int dstx, int dsty, int srcx, int srcy, int minx, int miny)
	{
		if (miny >= 0 && miny < dsty && srcy > 0)
		{
			for (auto y = 0; y < srcy; y++)
			{
				auto dstoffset = (miny + y) * dstx + minx;
				auto srcoffset = y * srcx;

				MemCopy(dst, dstoffset, src, srcoffset, srcx);
			}
		}
	}

	// Copy 3D[minx + x][miny + y][minz + z]
	static void Copy3D(double*& dst, double*& src, int dstx, int dsty, int dstz, int srcx, int srcy, int srcz, int minx, int miny, int minz)
	{
		if (minx >= 0 && minx < srcx && miny >= 0 && miny < srcy && minz >= 0 && minz < srcz)
		{
			for (auto z = 0; z < dstz; z++)
			{
				auto offsetd = z * dsty;
				auto offsets = (minz + z) * srcy + miny;

				for (auto y = 0; y < dsty; y++)
				{
					auto dstoffset = (offsetd + y) * dstx;
					auto srcoffset = (offsets + y) * srcx + minx;

					MemCopy(dst, dstoffset, src, srcoffset, dstx);
				}
			}
		}
	}

	// Copy 3D[x][y][index_list]
	static void Copy3DZ(double*& dst, double*& src, int*& index_list, int dstx, int dsty, int dstz, int srcx, int srcy, int srcz, int minz)
	{
		if (minz < srcz)
		{
			for (auto z = 0; z < dstz; z++)
			{
				auto zz = index_list[minz + z];

				for (auto y = 0; y < dsty; y++)
				{
					auto dstoffset = (z * dsty + y) * dstx;
					auto srcoffset = (zz * srcy + y) * srcx;

					MemCopy(dst, dstoffset, src, srcoffset, dstx);
				}
			}
		}
	}

	// Copies a 4D [index][x][y][z] to 3D [x][y][z]
	static void Copy4D3D(double*& dst, double*& src, int length, int index)
	{
		MemCopy(dst, 0, src, index * length, length);
	}

	// Copies a 3D [x][y][z] to 4D [index][x][y][z] with subsampling
	static void Copy3D4D(double*& dst, double*& src, int dstx, int dsty, int dstz, int srcx, int srcy, int srcz, int index, int step)
	{
		if (dstz == srcz)
		{
			for (auto z = 0; z < dstz; z++)
			{
				auto offsetd = index * dstz * dsty + z * dsty;
				auto offsets = z * srcy;

				for (auto y = 0; y < dsty; y++)
				{
					auto dstoffset = (offsetd + y) * dstx;
					auto srcoffset = (offsets + y * step) * srcx;

					for (auto x = 0; x < dstx; x++)
					{
						dst[dstoffset + x] = src[srcoffset + x * step];
					}
				}
			}
		}
	}

	// Copies a 3D [x][y][z] to 4D [index][x][y][z] with maxpooling
	static void Pool3D4D(double*& dst, double*& src, int dstx, int dsty, int dstz, int srcx, int srcy, int srcz, int index, int step)
	{
		if (dstz == srcz)
		{
			for (auto z = 0; z < dstz; z++)
			{
				auto offsetd = index * dstz * dsty + z * dsty;
				auto offsets = z * srcy;

				for (auto y = 0; y < dsty; y++)
				{
					auto dstoffset = (offsetd + y) * dstx;
					auto ys = y * step;

					for (auto x = 0; x < dstx; x++)
					{
						auto maxval = std::numeric_limits<double>::min();
						auto xs = x * step;

						for (auto yy = 0; yy < step; yy++)
						{
							auto dy = ys + yy;
							auto vstep = (offsets + dy) * srcx;

							for (auto xx = 0; xx < step; xx++)
							{
								auto dx = xs + xx;

								if (dx < srcx && dy < srcy)
								{
									auto val = src[vstep + dx];

									if (val > maxval)
									{
										maxval = val;
									}
								}
							}
						}

						dst[dstoffset + x] = maxval;
					}
				}
			}
		}
	}

	// Copies a 3D [x][y][z] to 4D [index][x][y][z]
	static void Copy3D4D(double*& dst, double*& src, int length, int index)
	{
		MemCopy(dst, index * length, src, 0, length);
	}

	// Copies a 2D [x][y] to 3D [index][x][y]
	static void Copy2D3D(double*& dst, double*& src, int dstx, int dsty, int dstz, int srcx, int srcy, int index)
	{
		auto size2D = srcx * srcy;

		if (index >= 0 && index < dstz && srcx == dstx && srcy == dsty)
		{
			auto dstoffset = index * size2D;

			for (auto y = 0; y < srcy; y++)
			{
				auto srcoffset = y * srcx;

				MemCopy(dst, dstoffset + srcoffset, src, srcoffset, srcx);
			}
		}
	}

	// Copies a 2D [x][y] to 4D [index][x][y][z]
	static void Copy2D4D(double*& dst, double*& src, int dstx, int dsty, int dstz, int srcx, int srcy, int z, int index)
	{
		auto size2D = srcx * srcy;
		auto size3D = size2D * dstz;

		if (index >= 0 && srcx == dstx && srcy == dsty)
		{
			auto dstoffset = index * size3D + z * size2D;

			for (auto y = 0; y < srcx; y++)
			{
				auto srcoffset = y * srcx;

				MemCopy(dst, srcoffset + dstoffset, src, srcoffset, srcx);
			}
		}
	}

	// Copies a 4D [index][x][y][z] to 2D [x][y] 
	static void Copy4D2D(double*& dst, double*& src, int dstx, int dsty, int srcx, int srcy, int srcz, int z, int index)
	{
		auto size2D = dstx * dsty;
		auto size3D = size2D * srcz;

		if (index >= 0 && srcx == dstx && srcy == dsty)
		{
			auto srcoffset = index * size3D + z * size2D;

			for (auto y = 0; y < dsty; y++)
			{
				auto dstoffset = y * dstx;

				MemCopy(dst, dstoffset, src, srcoffset + dstoffset, dstx);
			}
		}
	}

	// Copies a 4D [i][j][x][y] to a 2D [x][y] array
	static void Copy4DIJ2D(double*& dst, double*& src, int dstx, int dsty, int srcj, int srci, int i, int j)
	{
		auto size2D = dstx * dsty;
		auto srcoffset = (i * srcj + j) * size2D;

		if (j < srcj && i < srci)
		{
			MemCopy(dst, 0, src, srcoffset, size2D);
		}
	}

	// Copies a 2D [x][y] array to a 4D [i][j][x][y] 
	static void Copy2D4DIJ(double*& dst, double*& src, int dsty, int dstx, int srcx, int srcy, int i, int j)
	{
		auto size2D = srcx * srcy;
		auto dstoffset = (i * dstx + j) * size2D;

		if (j >= 0 && j < dstx && i >= 0 && i < dsty)
		{
			MemCopy(dst, dstoffset, src, 0, size2D);
		}
	}

	static void Free(double*& mem)
	{
		if (mem != NULL)
		{
			delete[] mem;
			mem = NULL;
		}
	}

	static void Free(int*& mem)
	{
		if (mem != NULL)
		{
			delete[] mem;
			mem = NULL;
		}
	}
};
#endif
