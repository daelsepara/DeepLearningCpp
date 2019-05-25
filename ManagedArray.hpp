#ifndef MANAGED_ARRAY_HPP
#define MANAGED_ARRAY_HPP

class ManagedIntList
{
private:

	int* Data = NULL;

	void _Free(int*& mem)
	{
		if (mem != NULL)
		{
			delete[] mem;
			mem = NULL;
		}
	}
	
	int* _IntList(int size)
	{
		auto temp = new int[size];

		for (auto i = 0; i < size; i++)
			temp[i] = i;

		return temp;
	}

	
public:

	int x = 0;

	ManagedIntList(int size)
	{
		x = size;

		Data = _IntList(size);
	}

	// 1D arrays
	int& operator()(int ix)
	{
		return Data[ix];
	}
	
	int Length()
	{
		return x;
	}
	
	void Free()
	{
		_Free(Data);

		x = 0;

		Data = NULL;
	}
};

class ManagedArray
{
private:

	double* Data = NULL;

	double* _New(int size, bool initialize = true)
	{
		auto temp = new double[size];

		if (initialize)
		{
			for (auto i = 0; i < size; i++)
				temp[i] = 0.0;
		}

		return temp;
	}
	
	void _Free(double*& mem)
	{
		if (mem != NULL)
		{
			delete[] mem;
			mem = NULL;
		}
	}

public:

	int x = 0;
	int y = 0;
	int z = 0;
	int i = 0;
	int j = 0;

	ManagedArray()
	{

	}

	ManagedArray(int size, bool initialize = true)
	{
		Resize(size, initialize);
	}

	ManagedArray(int sizex, int sizey, bool initialize = true)
	{
		Resize(sizex, sizey, initialize);
	}

	ManagedArray(int sizex, int sizey, int sizez, bool initialize = true)
	{
		Resize(sizex, sizey, sizez, initialize);
	}

	// For 4D arrays of type: [i][j] of [x][y] and [i] of [x][y][z]
	ManagedArray(int sizex, int sizey, int sizez, int sizei, int sizej, bool initialize = true)
	{
		Resize(sizex, sizey, sizez, sizei, sizej, initialize);
	}

	// 1D arrays
	double& operator()(int ix)
	{
		return Data[ix];
	}

	// 2D arrays
	double& operator()(int ix, int iy)
	{
		return Data[iy * x + ix];
	}

	// 3D arrays
	double& operator()(int ix, int iy, int iz)
	{
		return Data[(iz * y + iy) * x + ix];
	}

	void Resize(int size, bool initialize = true)
	{
		_Free(Data);

		x = size;
		y = 1;
		z = 1;
		i = 1;
		j = 1;

		Data = _New(size, initialize);
	}

	void Resize(int sizex, int sizey, bool initialize = true)
	{
		_Free(Data);

		x = sizex;
		y = sizey;
		z = 1;
		i = 1;
		j = 1;

		Data = _New(x * y, initialize);
	}

	void Resize(int sizex, int sizey, int sizez, bool initialize = true)
	{
		_Free(Data);

		x = sizex;
		y = sizey;
		z = sizez;
		i = 1;
		j = 1;

		Data = _New(x * y * z, initialize);
	}

	// For 4D arrays of type: [i][j] of [x][y] and [i] of [x][y][z]
	void Resize(int sizex, int sizey, int sizez, int sizei, int sizej, bool initialize = true)
	{
		_Free(Data);

		x = sizex;
		y = sizey;
		z = sizez;
		i = sizei;
		j = sizej;

		Data = _New(x * y * z * i * j, initialize);
	}

	int Length()
	{
		return x * y* z* i* j;
	}

	// Reshape without modifying data
	void Reshape(int ix = 1, int iy = 1, int iz = 1, int ii = 1, int ij = 1)
	{
		x = ix;
		y = iy;
		z = iz;
		i = ii;
		j = ij;
	}

	void Resize(ManagedArray& a, bool initialize = true)
	{
		_Free(Data);

		x = a.x;
		y = a.y;
		z = a.z;
		i = a.i;
		j = a.j;

		Data = _New(x * y * z * i * j, initialize);
	}
	
	void Free()
	{
		_Free(Data);

		x = 0;
		y = 0;
		z = 0;
		i = 0;
		j = 0;

		Data = NULL;
	}
};

#endif
