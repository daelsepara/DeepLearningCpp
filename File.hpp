#ifndef FILE_HPP
#define FILE_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>

#include "Matrix.hpp"
#include "MemOps.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define strdup _strdup
#endif

class File
{
public:

	static void Load1D(std::string filename, double*& A, int Ax)
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

		for (auto x = 0; x < Ax; x++)
		{
			A[x] = atof(token);

			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			
				token = strtok_s(NULL, ",", &next_token);
			
			#else
			
				token = strtok(NULL, ",");
			
			#endif
		}

		file.close();
	}

	static void Save1D(std::string filename, double*& A, int Ax)
	{
		std::ofstream file(filename);

		for (auto x = 0; x < Ax; x++)
		{
			file << A[x];

			if (x < Ax - 1)
			{
				file << ",";
			}
		}

		file << std::endl;

		file.close();
	}

	static void Load1DY(std::string filename, double*& A, int Ax)
	{
		std::ifstream file(filename); // open this file for input
		std::string line;

		for (auto x = 0; x < Ax; x++)
		{
			std::getline(file, line);

			A[x] = atof(line.c_str());
		}

		file.close();
	}

	static void Save1DY(std::string filename, double*& A, int Ax)
	{
		std::ofstream file(filename);

		for (auto x = 0; x < Ax; x++)
		{
			file << A[x] << std::endl;
		}

		file.close();
	}

	static void Load2D(std::string filename, double*& A, int Ax, int Ay)
	{
		auto temp = MemOps::New(Ax * Ay);

		std::ifstream file(filename);

		for (auto y = 0; y < Ay; y++)
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

			for (auto x = 0; x < Ax; x++)
			{
				MemOps::Set2D(temp, Ax, Ay, x, y, atof(token));

				#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

					token = strtok_s(NULL, ",", &next_token);

				#else

					token = strtok(NULL, ",");

				#endif
			}

			free(current_line);
		}

		MemOps::Copy2D(A, temp, Ax, Ay, Ax, Ay, 0, 0);

		MemOps::Free(temp);

		file.close();
	}

	static void Save2D(std::string filename, double*& A, int Ax, int Ay)
	{
		std::ofstream file(filename);

		for (auto y = 0; y < Ay; y++)
		{
			for (auto x = 0; x < Ax; x++)
			{
				file << MemOps::Val2D(A, Ax, Ay, x, y);

				if (x < Ax - 1)
				{
					file << ",";
				}
			}

			file << std::endl;
		}

		file.close();
	}

	static void Load2D4D(std::string filename, double*& A, int Ax, int Ay, int Ai, int Aj, int i, int j)
	{
		auto temp = MemOps::New(Ax * Ay);

		std::ifstream file(filename);

		for (auto y = 0; y < Ay; y++)
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

			for (auto x = 0; x < Ax; x++)
			{
				MemOps::Set2D(temp, Ax, Ay, x, y, atof(token));

				#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

					token = strtok_s(NULL, ",", &next_token);
				
				#else
					
					token = strtok(NULL, ",");
				
				#endif
			}

			free(current_line);
		}

		MemOps::Copy2D4DIJ(A, temp, Ai, Aj, Ax, Ay, i, j);

		MemOps::Free(temp);

		file.close();
	}

	static void Save2D4D(std::string filename, double*& A, int Ax, int Ay, int Ai, int Aj, int i, int j)
	{
		std::ofstream file(filename);

		auto temp = MemOps::New(Ax * Ay);

		MemOps::Copy4DIJ2D(temp, A, Ax, Ay, Aj, Ai, i, j);

		for (auto y = 0; y < Ay; y++)
		{
			for (auto x = 0; x < Ax; x++)
			{
				file << MemOps::Val2D(temp, Ax, Ay, x, y);

				if (x < Ax - 1)
				{
					file << ",";
				}
			}

			file << std::endl;
		}

		MemOps::Free(temp);

		file.close();
	}

	static void Load3D(std::string filename, double*& A, int Ax, int Ay, int Az)
	{
		std::ifstream file(filename);

		for (auto y = 0; y < Ay; y++)
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

			for (auto z = 0; z < Az; z++)
			{
				for (auto x = 0; x < Ax; x++)
				{
					if (token != NULL)
					{
						MemOps::Set3D(A, Ax, Ay, Az, x, y, z, atof(token));

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

	static void Save3D(std::string filename, double*& A, int Ax, int Ay, int Az)
	{
		std::ofstream file(filename);

		for (auto y = 0; y < Ay; y++)
		{
			for (auto z = 0; z < Az; z++)
			{
				for (auto x = 0; x < Ax; x++)
				{
					file << MemOps::Val3D(A, Ax, Ay, Az, x, y, z);

					if (z < Az - 1 || x < Ax - 1)
					{
						file << ",";
					}
				}
			}

			file << std::endl;
		}

		file.close();
	}
	
	static double* LoadData(std::string BaseDirectory, std::string BaseFileName, int sizex, int sizey, int sizez)
	{
		auto data = MemOps::New(sizex * sizey * sizez);

		char buffer[200];

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
		
				sprintf_s(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());
		
		#else
		
				sprintf(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());
		
		#endif

		std::string filename = buffer;

		File::Load3D(filename, data, sizex, sizey, sizez);

		return data;
	}

	static void SaveData(std::string BaseDirectory, std::string BaseFileName, double*& data, int sizex, int sizey, int sizez)
	{
		char buffer[200];

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

				sprintf_s(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#else

				sprintf(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#endif

		std::string filename = buffer;

		File::Save3D(filename, data, sizex, sizey, sizez);
	}

	static double* LoadClassification(std::string BaseDirectory, std::string BaseFileName, int sizex, int sizey)
	{
		auto classification = MemOps::New(sizex * sizey);

		char buffer[200];

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

			sprintf_s(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#else

			sprintf(buffer, "%s/%s.txt", BaseDirectory.c_str(), BaseFileName.c_str());

		#endif

		std::string filename = buffer;

		File::Load2D(filename, classification, sizex, sizey);

		return classification;
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

};
#endif
