#ifndef MANAGED_FILE_HPP
#define MANAGED_FILE_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>

#include "ManagedMatrix.hpp"
#include "ManagedOps.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define strdup _strdup
#endif

class ManagedFile
{
public:

	static void Load1D(std::string filename, ManagedArray& A)
	{
		std::ifstream file(filename); // open this file for input
		std::string line;

		std::getline(file, line);

		auto current_line = strdup(line.c_str());
		char* next_token = NULL;

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		
			auto token = strtok_s(current_line, ",", &next_token);
		
		#else

			auto token = strtok(current_line, ",");
			
		#endif

		for (auto x = 0; x < A.Length(); x++)
		{
			A(x) = atof(token);

			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

				token = strtok_s(NULL, ",", &next_token);

			#else
			
				token = strtok(NULL, ",");
			
			#endif
		}

		free(current_line);

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

			A(y) = atof(line.c_str());
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

			auto current_line = strdup(line.c_str());
			char* next_token = NULL;

			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
						
				auto token = strtok_s(current_line, ",", &next_token);

			#else
						
				auto token = strtok(current_line, ",");

			#endif

			for (auto x = 0; x < A.x; x++)
			{
				temp(x, y) = atof(token);

				#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

					token = strtok_s(NULL, ",", &next_token);

				#else

					token = strtok(NULL, ",");

				#endif
			}

			free(current_line);
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

			auto current_line = strdup(line.c_str());
			char* next_token = NULL;

			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			
				auto token = strtok_s(current_line, ",", &next_token);
			
			#else
			
				auto token = strtok(current_line, ",");
			
			#endif

			for (auto x = 0; x < A.x; x++)
			{
				temp(x, y) = atof(token);

				#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
				
					token = strtok_s(NULL, ",", &next_token);

				#else
				
					token = strtok(NULL, ",");
					
				#endif
			}

			free(current_line);
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

			auto current_line = strdup(line.c_str());
			char* next_token = NULL;

			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
						
				auto token = strtok_s(current_line, ",", &next_token);

			#else
						
				auto token = strtok(current_line, ",");

			#endif

			for (auto z = 0; z < A.z; z++)
			{
				for (auto x = 0; x < A.x; x++)
				{
					if (token != NULL)
					{
						A(x, y, z) = atof(token);

						#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
						
							token = strtok_s(NULL, ",", &next_token);
						
						#else
						
							token = strtok(NULL, ",");
						
						#endif
					}
				}
			}

			free(current_line);
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

			auto current_line = strdup(line.c_str());
			char* next_token = NULL;

			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			
				auto token = strtok_s(current_line, ",", &next_token);
			
			#else
			
				auto token = strtok(current_line, ",");
			
			#endif

			auto xoffset = y * xx;

			for (auto z = 0; z < zz; z++)
			{
				auto yoffset = z * size2D;

				for (auto x = 0; x < xx; x++)
				{
					if (token != NULL)
					{
						A(xoffset + x, yoffset) = atof(token);

						#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
						
							token = strtok_s(NULL, ",", &next_token);
						
						#else
						
							token = strtok(NULL, ",");
						
						#endif
					}
				}
			}

			free(current_line);
		}

		file.close();
	}

	static void SaveJSON(std::string BaseDirectory, std::string BaseFileName, std::string json_string)
	{
		char buffer[200];

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		
			sprintf_s(buffer, "%s/%s.json", BaseDirectory.c_str(), BaseFileName.c_str());

		#else
		
			sprintf(buffer, "%s/%s.json", BaseDirectory.c_str(), BaseFileName.c_str());
		
		#endif

		std::string filename = buffer;
		std::ofstream file(filename);

		file << json_string;

		file.close();
	}
	
	static ManagedArray LoadData(std::string BaseDirectory, std::string BaseFileName, int sizex, int sizey, int sizez)
	{
		auto data = ManagedArray(sizex, sizey, sizez);

		char buffer[200];

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		
			sprintf_s(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#else
		
			sprintf(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#endif

		std::string filename = buffer;

		Load3D(filename, data);

		return data;
	}

	static void SaveData(std::string BaseDirectory, std::string BaseFileName, ManagedArray& data)
	{
		char buffer[200];

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
				
			sprintf_s(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#else
				
			sprintf(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#endif

		std::string filename = buffer;

		Save3D(filename, data);
	}

	static ManagedArray LoadClassification(std::string BaseDirectory, std::string BaseFileName, int sizex, int sizey)
	{
		auto classification = ManagedArray(sizex, sizey);

		char buffer[200];

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
				
			sprintf_s(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());
		
		#else
				
			sprintf(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());
		
		#endif

		std::string filename = buffer;

		Load2D(filename, classification);

		return classification;
	}

	static void SaveClassification(std::string BaseDirectory, std::string BaseFileName, ManagedIntList classification)
	{
		char buffer[200];

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		
			sprintf_s(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#else
		
			sprintf(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());
		
		#endif

		std::string filename = buffer;
		std::ofstream file(filename);

		for (auto y = 0; y < classification.Length(); y++)
		{
			file << classification(y) << std::endl;
		}

		file.close();
	}
};
#endif
