#ifndef MANAGED_FILE_HPP
#define MANAGED_FILE_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "ManagedMatrix.hpp"
#include "ManagedOps.hpp"

class ManagedFile
{
public:

	static void Load1D(std::string filename, ManagedArray& A)
	{
		std::ifstream file(filename); // open this file for input

		std::string line;
		std::getline(file, line);

		std::istringstream is(line);
		std::string token;
			
		for (auto x = 0; x < A.Length(); x++)
		{
			std::getline(is, token, ',');

			A(x) = std::stod(token);
		}

		file.close();
	}

	static void Save1D(std::string filename, ManagedArray& A)
	{
		std::ofstream file(filename);

		for (auto x = 0; x < A.x; x++)
		{
			file << A(x);

			if (x < A.x - 1)
			{
				file << ",";
			}
		}

		file << std::endl;

		file.close();
	}

	static void Load1DY(std::string filename, ManagedArray& A)
	{
		std::ifstream file(filename); // open this file for input
		std::string line;

		for (auto y = 0; y < A.Length(); y++)
		{
			std::getline(file, line);

			A(y) = std::stod(line);
		}

		file.close();
	}

	static void Save1DY(std::string filename, ManagedArray& A)
	{
		std::ofstream file(filename);

		for (auto y = 0; y < A.Length(); y++)
		{
			file << A(y) << std::endl;
		}

		file.close();
	}

	static void Load2D(std::string filename, ManagedArray& A)
	{
		auto temp = ManagedArray(A.x, A.y);

		std::ifstream file(filename);

		for (auto y = 0; y < A.y; y++)
		{
			std::string line;
			std::getline(file, line);

			std::istringstream is(line);
			std::string token;
			
			for (auto x = 0; x < A.x; x++)
			{
				std::getline(is, token, ',');

				temp(x, y) = std::stod(token);
			}
		}

		ManagedOps::Copy2D(A, temp, 0, 0);

		ManagedOps::Free(temp);

		file.close();
	}

	static void Save2D(std::string filename, ManagedArray& A)
	{
		std::ofstream file(filename);

		for (auto y = 0; y < A.y; y++)
		{
			for (auto x = 0; x < A.x; x++)
			{
				file << A(x, y);

				if (x < A.x - 1)
				{
					file << ",";
				}
			}

			file << std::endl;
		}

		file.close();
	}

	static void Load2D4D(std::string filename, ManagedArray& A, int i, int j)
	{
		auto temp = ManagedArray(A.x, A.y);

		std::ifstream file(filename);

		for (auto y = 0; y < A.y; y++)
		{
			std::string line;
			std::getline(file, line);

			std::istringstream is(line);
			std::string token;

			for (auto x = 0; x < A.x; x++)
			{
				std::getline(is, token, ',');

				temp(x, y) = std::stod(token);
			}
		}

		ManagedOps::Copy2D4DIJ(A, temp, i, j);

		ManagedOps::Free(temp);

		file.close();
	}

	static void Save2D4D(std::string filename, ManagedArray& A, int i, int j)
	{
		std::ofstream file(filename);

		auto temp = ManagedArray(A.x, A.y);

		ManagedOps::Copy4DIJ2D(temp, A, i, j);

		for (auto y = 0; y < A.y; y++)
		{
			for (auto x = 0; x < A.x; x++)
			{
				file << temp(x, y);

				if (x < A.x - 1)
				{
					file << ",";
				}
			}

			file << std::endl;
		}

		ManagedOps::Free(temp);

		file.close();
	}

	static void Load3D(std::string filename, ManagedArray& A)
	{
		std::ifstream file(filename);

		for (auto y = 0; y < A.y; y++)
		{
			std::string line;
			std::getline(file, line);

			std::istringstream is(line);
			std::string token;

			for (auto z = 0; z < A.z; z++)
			{
				for (auto x = 0; x < A.x; x++)
				{
					if (std::getline(is, token, ','))
					{
						A(x, y, z) = std::stod(token);
					}
				}
			}
		}

		file.close();
	}

	static void Save3D(std::string filename, ManagedArray& A)
	{
		std::ofstream file(filename);

		for (auto y = 0; y < A.y; y++)
		{
			for (auto z = 0; z < A.z; z++)
			{
				for (auto x = 0; x < A.x; x++)
				{
					file << A(x, y, z);

					if (z < A.z - 1 || x < A.x - 1)
					{
						file << ",";
					}
				}
			}

			file << std::endl;
		}

		file.close();
	}

	// Load 3D data from a file and reshape into 2D, i.e. (y, x:z) -> (z, x:y)
	static void Load3D2D(std::string filename, ManagedArray& A)
	{
		std::ifstream file(filename);

		auto xx = A.x;
		auto yy = A.y;
		auto zz = A.z;
		auto size2D = xx * yy;

		A.Reshape(xx * yy, zz);

		for (auto y = 0; y < yy; y++)
		{
			std::string line;
			std::getline(file, line);

			std::istringstream is(line);
			std::string token;

			auto xoffset = y * xx;

			for (auto z = 0; z < zz; z++)
			{
				auto yoffset = z * size2D;

				for (auto x = 0; x < xx; x++)
				{
					if (std::getline(is, token, ','))
					{
						A(xoffset + x, yoffset) = std::stod(token);
					}
				}
			}
		}

		file.close();
	}

	static void SaveJSON(std::string BaseDirectory, std::string BaseFileName, std::string json_string)
	{
		std::ostringstream buffer;

		buffer << BaseDirectory << "/" << BaseFileName << ".json";

		std::string filename = buffer.str();
		
		std::ofstream file(filename);

		file << json_string;

		file.close();
	}
	
	static ManagedArray LoadData(std::string BaseDirectory, std::string BaseFileName, int sizex, int sizey, int sizez)
	{
		auto data = ManagedArray(sizex, sizey, sizez);

		std::ostringstream buffer;

		buffer << BaseDirectory << "/" << BaseFileName << ".txt";

		std::string filename = buffer.str();

		Load3D(filename, data);

		return data;
	}

	static void SaveData(std::string BaseDirectory, std::string BaseFileName, ManagedArray& data)
	{
		std::ostringstream buffer;

		buffer << BaseDirectory << "/" << BaseFileName << ".txt";

		std::string filename = buffer.str();

		Save3D(filename, data);
	}

	static ManagedArray LoadClassification(std::string BaseDirectory, std::string BaseFileName, int sizex, int sizey)
	{
		auto classification = ManagedArray(sizex, sizey);

		std::ostringstream buffer;

		buffer << BaseDirectory << "/" << BaseFileName << ".txt";

		std::string filename = buffer.str();

		Load2D(filename, classification);

		return classification;
	}

	static void SaveClassification(std::string BaseDirectory, std::string BaseFileName, ManagedIntList classification)
	{
		std::ostringstream buffer;

		buffer << BaseDirectory << "/" << BaseFileName << ".txt";

		std::string filename = buffer.str();

		std::ofstream file(filename);

		for (auto y = 0; y < classification.Length(); y++)
		{
			file << classification(y) << std::endl;
		}

		file.close();
	}
};
#endif
