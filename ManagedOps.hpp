#ifndef MANAGED_OPS_HPP
#define MANAGED_OPS_HPP

#include "ManagedArray.hpp"

class ManagedOps
{
public:

	static void MemCopy(ManagedArray& dst, int dstoffset, ManagedArray& src, int srcoffset, int count)
	{
		for (auto i = 0; i < count; i++)
			dst(dstoffset + i) = src(srcoffset + i);
	}

	static void Set(ManagedArray& dst, double value)
	{
		for (auto i = 0; i < dst.Length(); i++)
			dst(i) = value;
	}

	static void Set(ManagedIntList& dst, int value)
	{
		for (auto i = 0; i < dst.Length(); i++)
			dst(i) = value;
	}

	// Copy 2D[minx + x][miny + y]
	static void Copy2D(ManagedArray& dst, ManagedArray& src, int minx, int miny)
	{
		if (miny >= 0 && miny < src.y)
		{
			for (auto y = 0; y < dst.y; y++)
			{
				auto dstoffset = y * dst.x;
				auto srcoffset = (miny + y) * src.x + minx;

				MemCopy(dst, dstoffset, src, srcoffset, dst.x);
			}
		}
	}

	// Copy 2D[index_list][y]
	static void Copy2DX(ManagedArray& dst, ManagedArray& src, ManagedIntList& index_list, int minx)
	{
		for (auto y = 0; y < dst.y; y++)
		{
			auto dstoffset = y * dst.x;
			auto srcoffset = y * src.x;

			auto xx = index_list(minx);

			MemCopy(dst, dstoffset, src, srcoffset + xx, dst.x);
		}
	}

	// Copy 2D[x][y] to 2D[minx + x][miny + y]
	static void Copy2DOffset(ManagedArray& dst, ManagedArray& src, int minx, int miny)
	{
		if (miny >= 0 && miny < dst.y && src.y > 0)
		{
			for (auto y = 0; y < src.y; y++)
			{
				auto dstoffset = (miny + y) * dst.x + minx;
				auto srcoffset = y * src.x;

				MemCopy(dst, dstoffset, src, srcoffset, src.x);
			}
		}
	}

	// Copy 3D[minx + x][miny + y][minz + z]
	static void Copy3D(ManagedArray& dst, ManagedArray& src, int minx, int miny, int minz)
	{
		if (minx >= 0 && minx < src.x && miny >= 0 && miny < src.y && minz >= 0 && minz < src.z)
		{
			for (auto z = 0; z < dst.z; z++)
			{
				auto offsetd = z * dst.y;
				auto offsets = (minz + z) * src.y + miny;

				for (auto y = 0; y < dst.y; y++)
				{
					auto dstoffset = (offsetd + y) * dst.x;
					auto srcoffset = (offsets + y) * src.x + minx;

					MemCopy(dst, dstoffset, src, srcoffset, dst.x);
				}
			}
		}
	}

	// Copy 3D[x][y][index_list]
	static void Copy3DZ(ManagedArray& dst, ManagedArray& src, ManagedIntList& index_list, int minz)
	{
		if (minz < src.z)
		{
			for (auto z = 0; z < dst.z; z++)
			{
				auto zz = index_list(minz + z);

				for (auto y = 0; y < dst.y; y++)
				{
					auto dstoffset = (z * dst.y + y) * dst.x;
					auto srcoffset = (zz * src.y + y) * src.x;

					MemCopy(dst, dstoffset, src, srcoffset, dst.x);
				}
			}
		}
	}

	// Copies a 4D [index][x][y][z] to 3D [x][y][z]
	static void Copy4D3D(ManagedArray& dst, ManagedArray& src, int index)
	{
		MemCopy(dst, 0, src, index * dst.Length(), dst.Length());
	}

	// Copies a 3D [x][y][z] to 4D [index][x][y][z] with subsampling
	static void Copy3D4D(ManagedArray& dst, ManagedArray& src, int index, int step)
	{
		if (dst.z == src.z)
		{
			for (auto z = 0; z < dst.z; z++)
			{
				auto offsetd = index * dst.z * dst.y + z * dst.y;
				auto offsets = z * src.y;

				for (auto y = 0; y < dst.y; y++)
				{
					auto dstoffset = (offsetd + y) * dst.x;
					auto srcoffset = (offsets + y * step) * src.x;

					for (auto x = 0; x < dst.x; x++)
					{
						dst(dstoffset + x) = src(srcoffset + x * step);
					}
				}
			}
		}
	}

	// Copies a 3D [x][y][z] to 4D [index][x][y][z] with maxpooling
	static void Pool3D4D(ManagedArray& dst, ManagedArray& src, int index, int step)
	{
		if (dst.z == src.z)
		{
			for (auto z = 0; z < dst.z; z++)
			{
				auto offsetd = index * dst.z * dst.y + z * dst.y;
				auto offsets = z * src.y;

				for (auto y = 0; y < dst.y; y++)
				{
					auto dstoffset = (offsetd + y) * dst.x;
					auto ys = y * step;

					for (auto x = 0; x < dst.x; x++)
					{
						auto maxval = std::numeric_limits<double>::min();
						auto xs = x * step;

						for (auto yy = 0; yy < step; yy++)
						{
							auto dy = ys + yy;
							auto vstep = (offsets + dy) * src.x;

							for (auto xx = 0; xx < step; xx++)
							{
								auto dx = xs + xx;

								if (dx < src.x && dy < src.y)
								{
									auto val = src(vstep + dx);

									if (val > maxval)
										maxval = val;
								}
							}
						}

						dst(dstoffset + x) = maxval;
					}
				}
			}
		}
	}

	// Copies a 3D [x][y][z] to 4D [index][x][y][z]
	static void Copy3D4D(ManagedArray& dst, ManagedArray& src, int index)
	{
		MemCopy(dst, index * src.Length(), src, 0, src.Length());
	}

	// Copies a 2D [x][y] to 3D [index][x][y]
	static void Copy2D3D(ManagedArray& dst, ManagedArray& src, int index)
	{
		auto size2D = src.x * src.y;

		if (index >= 0 && index < dst.z && src.x == dst.x && src.y == dst.y)
		{
			auto dstoffset = index * size2D;

			for (auto y = 0; y < src.y; y++)
			{
				auto srcoffset = y * src.x;

				MemCopy(dst, dstoffset + srcoffset, src, srcoffset, src.x);
			}
		}
	}

	// Copies a 2D [x][y] to 4D [index][x][y][z]
	static void Copy2D4D(ManagedArray& dst, ManagedArray& src, int z, int index)
	{
		auto size2D = src.x * src.y;
		auto size3D = size2D * dst.z;

		if (index >= 0 && src.x == dst.x && src.y == dst.y)
		{
			auto dstoffset = index * size3D + z * size2D;

			for (auto y = 0; y < src.x; y++)
			{
				auto srcoffset = y * src.x;

				MemCopy(dst, srcoffset + dstoffset, src, srcoffset, src.x);
			}
		}
	}

	// Copies a 4D [index][x][y][z] to 2D [x][y] 
	static void Copy4D2D(ManagedArray& dst, ManagedArray& src, int z, int index)
	{
		auto size2D = dst.x * dst.y;
		auto size3D = size2D * src.z;

		if (index >= 0 && src.x == dst.x && src.y == dst.y)
		{
			auto srcoffset = index * size3D + z * size2D;

			for (auto y = 0; y < dst.y; y++)
			{
				auto dstoffset = y * dst.x;

				MemCopy(dst, dstoffset, src, srcoffset + dstoffset, dst.x);
			}
		}
	}

	// Copies a 4D [i][j][x][y] to a 2D [x][y] array
	static void Copy4DIJ2D(ManagedArray& dst, ManagedArray& src, int i, int j)
	{
		auto size2D = dst.x * dst.y;
		auto srcoffset = (i * src.j + j) * size2D;

		if (j < src.j && i < src.i)
		{
			MemCopy(dst, 0, src, srcoffset, size2D);
		}
	}

	// Copies a 2D [x][y] array to a 4D [i][j][x][y] 
	static void Copy2D4DIJ(ManagedArray& dst, ManagedArray& src, int i, int j)
	{
		auto size2D = src.x * src.y;
		auto dstoffset = (i * dst.j + j) * size2D;

		if (j >= 0 && j < dst.j && i >= 0 && i < dst.i)
		{
			MemCopy(dst, dstoffset, src, 0, size2D);
		}
	}

	static void Free(ManagedArray& item)
	{
		item.Free();
	}

	static void Free(ManagedIntList& item)
	{
		item.Free();
	}
};

#endif
