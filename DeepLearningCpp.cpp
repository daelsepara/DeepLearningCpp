#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ManagedCNN.hpp"
#include "ManagedDNN.hpp"
#include "ManagedNN.hpp"
#include "OptimizerDNN.hpp"
#include "OptimizerNN.hpp"

#include "ManagedUtil.hpp"
#include "Profiler.hpp"
#include "Random.hpp"
#include "ManagedUtil.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define strdup _strdup
#endif

void ParseInt(std::string arg, const char* str, const char* var, int& dst)
{
	auto len = strlen(str);

	if (len > 0)
	{
		if (!arg.compare(0, len, str) && arg.length() > len)
		{
			try
			{
				auto val = stoi(arg.substr(len));

				fprintf(stderr, "... %s = %d\n", var, val);

				dst = val;
			}
			catch (const std::invalid_argument & ia)
			{
				fprintf(stderr, "... %s = NaN %s\n", var, ia.what());
				exit(1);
			}
		}
	}
}

void ParseInts(std::string arg, const char* str, const char* var, std::vector<int> & ints)
{
	auto len = strlen(str);

	if (len > 0)
	{
		if (!arg.compare(0, len, str) && arg.length() > len)
		{
			try
			{
				auto s = arg.substr(len);

				std::string delimiter = ",";

				size_t pos = 0;

				while ((pos = s.find(delimiter)) != std::string::npos) {

					auto val = stoi(s.substr(0, pos));

					ints.push_back(val);

					s.erase(0, pos + delimiter.length());
				}

				if (s.length() > 0)
				{
					auto val = stoi(s.substr(0, pos));

					ints.push_back(val);
				}
			}
			catch (const std::invalid_argument & ia)
			{
				fprintf(stderr, "... %s = NaN %s\n", var, ia.what());
				exit(1);
			}
		}
	}
}

void ParseDouble(std::string arg, const char* str, const char* var, double& dst)
{
	auto len = strlen(str);

	if (len > 0)
	{
		if (!arg.compare(0, len, str) && arg.length() > len)
		{
			try
			{
				auto val = stod(arg.substr(len));

				fprintf(stderr, "... %s = %e\n", var, val);

				dst = val;
			}
			catch (const std::invalid_argument & ia)
			{
				fprintf(stderr, "... %s = NaN %s\n", var, ia.what());

				exit(1);
			}
		}
	}
}

void Load2D(std::string filename, ManagedArray& input, ManagedArray& output, const char* delimiter, int& inputs, int& categories, int& examples)
{
	auto temp = std::vector<std::vector<double>>();

	std::ifstream file(filename);
	std::string line;
	
	categories = 0;
	inputs = 0;
	examples = 0;
	
	while (std::getline(file, line))
    {
		if (std::strlen(line.c_str()) > 0)
		{
			temp.push_back(std::vector<double>());
			
			auto current_line = strdup(line.c_str());
			
			char* next_token = NULL;

			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			
				auto token = std::strtok_s(current_line, delimiter, &next_token);
			
			#else
			
				auto token = std::strtok(current_line, delimiter);
			
			#endif
			
			int tokens = 0;
			
			while (token != NULL)
			{
				tokens++;
				
				auto value = atof(token);
				
				temp[examples].push_back(value);
				
				#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
				
					token = std::strtok_s(NULL, delimiter, &next_token);
				
				#else
			
					token = strtok(NULL, delimiter);
			
				#endif
			}
		
			free(current_line);
		
			if (tokens > 0)
				examples++;
		}
    }

	auto sizey = (int)temp.size();
	auto sizex = (int)temp[0].size();
	inputs = sizex - 1;

	input.Resize(inputs, sizey, false);
	output.Resize(1, sizey, false);
	
	for (auto y = 0; y < sizey; y++)
	{
		for (auto x = 0; x < sizex; x++)
		{
			auto category = inputs > 0 ? (int)temp[y][inputs] : 0;
			
			categories = std::max(categories, category);
			
			if (x < inputs)
				input(x, y) = temp[y][x];
				
			output(y) = category;
		}
	}
	
	file.close();
	

}

void DNNOptimizer(std::string InputData, int delimiter, double alpha, int epochs, double tolerance, std::vector<int> layers, bool save, std::string SaveDirectory, std::string SaveJSON)
{
	std::string BaseDirectory = "./";
	
	if (std::strlen(InputData.c_str()) > 0)
	{
		auto Inputs = 0;
		auto Categories = 0;
		auto Examples = 0;
		
		auto input = ManagedArray();
		auto output = ManagedArray();
		
		Load2D(InputData, input, output, delimiter == 0 ? "\t" : ",", Inputs, Categories, Examples);
		
		fprintf(stderr, "\n%d lines read with %d inputs and %d categories\n", Examples, Inputs, Categories);
		
		if (Inputs > 0 && Categories > 0 && Examples > 0 && layers.size() > 0)
		{
			auto opts = NeuralNetworkOptions(alpha, epochs, Categories, Inputs, Examples, tolerance, (int)layers.size(), false);
			
			auto dnn = OptimizerDNN();

			auto start = Profiler::now();

			printf("\nOptimizing Network...\n");

			dnn.SetupHiddenLayers(opts.Inputs, opts.Categories, layers);

			auto normalized_input = dnn.Normalize(input);
			auto normalized_test = dnn.ApplyNormalization(input);

			dnn.Run(normalized_input, output, opts);

			printf("Optimizitation Done\n");

			printf("elapsed time is %ld ms\n", Profiler::Elapsed(start));

			printf("Final Training error is %e\n", dnn.Cost);
			printf("L2 error is %e\n", dnn.L2);
			printf("Number of iterations: %d\n", dnn.Iterations);

			printf("\nNetwork Weights:\n");
			for (auto layer = 0; layer < (int)dnn.Weights.size(); layer++)
			{
				if (layer < (int)dnn.Weights.size() - 1)
				{
					printf("Layer %d:\n", layer);
				}
				else
				{
					printf("Output layer %d\n", layer);
				}
				
				ManagedMatrix::Print2D(dnn.Weights[layer]);
			}

			printf("\nClassifying...\n");

			start = Profiler::now();

			auto classification = dnn.Classify(normalized_test, opts, 0.9);

			printf("Classification Done\n");

			printf("elapsed time is %ld ms\n", Profiler::Elapsed(start));

			if (save && std::strlen(SaveJSON.c_str()) > 0)
			{
				printf("\nSaving Deep Neural Network Parameters\n");

				ManagedFile::SaveJSON(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, SaveJSON, ManagedUtil::Serialize(dnn));
			}

			dnn.Free();

			ManagedOps::Free(classification);
			ManagedOps::Free(normalized_input);
			ManagedOps::Free(normalized_test);
		}
		
		ManagedOps::Free(input);
		ManagedOps::Free(output);
	}
}

void DNNTrainer(std::string InputData, int delimiter, double alpha, int epochs, double tolerance, std::vector<int> layers, bool useL2, bool save, std::string SaveDirectory, std::string SaveJSON)
{
	std::string BaseDirectory = "./";
	
	if (std::strlen(InputData.c_str()) > 0)
	{
		auto Inputs = 0;
		auto Categories = 0;
		auto Examples = 0;
		
		auto input = ManagedArray();
		auto output = ManagedArray();
		
		Load2D(InputData, input, output, delimiter == 0 ? "\t" : ",", Inputs, Categories, Examples);
		
		fprintf(stderr, "\n%d lines read with %d inputs and %d categories\n", Examples, Inputs, Categories);
		
		if (Inputs > 0 && Categories > 0 && Examples > 0 && layers.size() > 0)
		{
			auto opts = NeuralNetworkOptions(alpha, epochs, Categories, Inputs, Examples, tolerance, (int)layers.size(), useL2);
			
			auto dnn = ManagedDNN();
			
			auto start = Profiler::now();

			printf("\nTraining Network...\n");

			dnn.SetupHiddenLayers(opts.Inputs, opts.Categories, layers);

			auto normalized_input = dnn.Normalize(input);
			auto normalized_test = dnn.ApplyNormalization(input);

			dnn.Train(normalized_input, output, opts);

			printf("Training Done\n");

			printf("elapsed time is %ld ms\n", Profiler::Elapsed(start));

			printf("Final Training error is %e\n", dnn.Cost);
			printf("L2 error is %e\n", dnn.L2);
			printf("Number of iterations: %d\n", dnn.Iterations);

			printf("\nNetwork Weights:\n");
			for (auto layer = 0; layer < (int)dnn.Weights.size(); layer++)
			{
				if (layer < (int)dnn.Weights.size() - 1)
				{
					printf("Layer %d:\n", layer);
				}
				else
				{
					printf("Output layer %d\n", layer);
				}
				
				ManagedMatrix::Print2D(dnn.Weights[layer]);
			}

			printf("\nClassifying...\n");

			start = Profiler::now();

			auto classification = dnn.Classify(normalized_test, opts, 0.9);

			printf("Classification Done\n");

			printf("elapsed time is %ld ms\n", Profiler::Elapsed(start));

			if (save && std::strlen(SaveJSON.c_str()) > 0)
			{
				printf("\nSaving Deep Neural Network Parameters\n");

				ManagedFile::SaveJSON(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, SaveJSON, ManagedUtil::Serialize(dnn));
			}

			dnn.Free();

			ManagedOps::Free(classification);
			ManagedOps::Free(normalized_input);
			ManagedOps::Free(normalized_test);
		}
		
		ManagedOps::Free(input);
		ManagedOps::Free(output);
	}
}

void NNTrainer(std::string InputData, int delimiter, double alpha, int Nodes, int epochs, double tolerance, bool useL2, bool save, std::string SaveDirectory, std::string SaveJSON)
{
	std::string BaseDirectory = "./";
	
	if (std::strlen(InputData.c_str()) > 0)
	{
		auto Inputs = 0;
		auto Categories = 0;
		auto Examples = 0;
		
		auto input = ManagedArray();
		auto output = ManagedArray();
		
		Load2D(InputData, input, output, delimiter == 0 ? "\t" : ",", Inputs, Categories, Examples);
		
		fprintf(stderr, "\n%d lines read with %d inputs and %d categories\n", Examples, Inputs, Categories);
		
		if (Inputs > 0 && Categories > 0 && Examples > 0 && Nodes > Inputs)
		{	
			auto opts = NeuralNetworkOptions(alpha, epochs, Categories, Inputs, Nodes, Examples, tolerance, useL2);
			
			auto nn = ManagedNN();
			
			auto start = Profiler::now();

			printf("\nTraining Network...\n");

			auto normalized_input = nn.Normalize(input);
			auto normalized_test = nn.ApplyNormalization(input);

			nn.Train(normalized_input, output, opts);

			printf("Training Done\n");

			printf("elapsed time is %ld ms\n", Profiler::Elapsed(start));

			printf("Final Training error is %e\n", nn.Cost);
			printf("L2 error is %e\n", nn.L2);
			printf("Number of iterations: %d\n", nn.Iterations);

			printf("\nNetwork Weights:\n");
			printf("Weights Wji:\n");
			ManagedMatrix::Print2D(nn.Wji);
			printf("Weights Wkj:\n");
			ManagedMatrix::Print2D(nn.Wkj);

			printf("\nClassifying...\n");

			start = Profiler::now();

			auto classification = nn.Classify(normalized_test, opts, 0.9);

			printf("Classification Done\n");

			printf("elapsed time is %ld ms\n", Profiler::Elapsed(start));

			if (save && std::strlen(SaveJSON.c_str()) > 0)
			{
				printf("\nSaving Deep Neural Network Parameters\n");

				ManagedFile::SaveJSON(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, SaveJSON, ManagedUtil::Serialize(nn));
			}

			nn.Free();

			ManagedOps::Free(classification);
			ManagedOps::Free(normalized_input);
			ManagedOps::Free(normalized_test);
		}
		
		ManagedOps::Free(input);
		ManagedOps::Free(output);
	}
}

void NNOptimizer(std::string InputData, int delimiter, double alpha, int Nodes, int epochs, double tolerance, bool save, std::string SaveDirectory, std::string SaveJSON)
{
	std::string BaseDirectory = "./";
	
	if (std::strlen(InputData.c_str()) > 0)
	{
		auto Inputs = 0;
		auto Categories = 0;
		auto Examples = 0;
		
		auto input = ManagedArray();
		auto output = ManagedArray();
		
		Load2D(InputData, input, output, delimiter == 0 ? "\t" : ",", Inputs, Categories, Examples);
		
		fprintf(stderr, "\n%d lines read with %d inputs and %d categories\n", Examples, Inputs, Categories);
		
		if (Inputs > 0 && Categories > 0 && Examples > 0 && Nodes > Inputs)
		{	
			auto opts = NeuralNetworkOptions(alpha, epochs, Categories, Inputs, Nodes, Examples, tolerance);
			
			auto nn = OptimizerNN();
			
			auto start = Profiler::now();

			printf("\nOptimizing Network...\n");

			auto normalized_input = nn.Normalize(input);
			auto normalized_test = nn.ApplyNormalization(input);

			nn.Run(normalized_input, output, opts);

			printf("Optimization Done\n");

			printf("elapsed time is %ld ms\n", Profiler::Elapsed(start));

			printf("Final Training error is %e\n", nn.Cost);
			printf("L2 error is %e\n", nn.L2);
			printf("Number of iterations: %d\n", nn.Iterations);

			printf("\nNetwork Weights:\n");
			printf("Weights Wji:\n");
			ManagedMatrix::Print2D(nn.Wji);
			printf("Weights Wkj:\n");
			ManagedMatrix::Print2D(nn.Wkj);

			printf("\nClassifying...\n");

			start = Profiler::now();

			auto classification = nn.Classify(normalized_test, opts, 0.9);

			printf("Classification Done\n");

			printf("elapsed time is %ld ms\n", Profiler::Elapsed(start));

			if (save && std::strlen(SaveJSON.c_str()) > 0)
			{
				printf("\nSaving Deep Neural Network Parameters\n");

				ManagedFile::SaveJSON(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, SaveJSON, ManagedUtil::Serialize(nn));
			}

			nn.Free();

			ManagedOps::Free(classification);
			ManagedOps::Free(normalized_input);
			ManagedOps::Free(normalized_test);
		}
		
		ManagedOps::Free(input);
		ManagedOps::Free(output);
	}
}

int main(int argc, char** argv)
{
	// convolutional neural network parameters
	auto shuffle = false;
	auto batchsize = 2;
	auto Pool = false;

	// neural network parameters
	auto nodes = 4;
	auto tolerance = 0.0001;
	
	// deep neural networks layer configuration
	std::vector<int> Layers;

	// common parameters
	auto alpha = 1.0;
	auto epochs = 10000;
	auto save = false;
	auto useL2 = false; // use L2 error on neural networks or deep neural networks
	
	char SaveDirectory[200];
	SaveDirectory[0] = '\0';

	std::string SaveDir;

	char SaveJSON[200];
	SaveJSON[0] = '\0';

	char InputData[200];
	InputData[0] = '\0';
	
	int delimiter = 0;
	
	auto RunDNNOptimizer = false;
	auto RunDNNTrainer = false;
	auto RunNNOptimizer = false;
	auto RunNNTrainer = false;
	
	for (auto i = 0; i < argc; i++)
	{
		std::string arg = argv[i];
		std::transform(arg.begin(), arg.end(), arg.begin(), ::toupper);

		if (!arg.compare("/SHUFFLE"))
		{
			shuffle = true;
		}
		else if (!arg.compare("/SAVE"))
		{
			save = true;
		}
		else if (!arg.compare("/POOL"))
		{
			Pool = true;
		}
		else if (!arg.compare("/USEL2"))
		{
			useL2 = true;
		}
		else if (!arg.compare("/TAB"))
		{
			delimiter = 0;
		}
		else if (!arg.compare("/COMMA"))
		{
			delimiter = 1;
		}
		else if (!arg.compare("/DNN"))
		{
			RunDNNTrainer = true;
		}
		else if (!arg.compare("/DMIN"))
		{
			RunDNNOptimizer = true;
		}
		else if (!arg.compare("/NN"))
		{
			RunNNTrainer = true;
		}
		else if (!arg.compare("/NMIN"))
		{
			RunNNOptimizer = true;
		}
		
		if (!arg.compare(0, 9, "/SAVEDIR=") && arg.length() > 9)
		{
			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

				strncpy_s(SaveDirectory, &argv[i][9], sizeof(SaveDirectory));
			
			#else

				strncpy(SaveDirectory, &argv[i][9], sizeof(SaveDirectory));

			#endif
		}

		if (!arg.compare(0, 6, "/JSON=") && arg.length() > 6)
		{
			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			
			strncpy_s(SaveJSON, &argv[i][6], sizeof(SaveJSON));

			#else
				
				strncpy(SaveJSON, &argv[i][6], sizeof(SaveJSON));

			#endif
		}
		
		if (!arg.compare(0, 7, "/INPUT=") && arg.length() > 7)
		{
			#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
			
			strncpy_s(InputData, &argv[i][7], sizeof(InputData));

			#else
				
				strncpy(InputData, &argv[i][7], sizeof(InputData));

			#endif
		}

		ParseInt(arg, "/BATCH=", "Batch Size", batchsize);
		ParseInt(arg, "/EPOCH=", "Epochs", epochs);
		ParseInt(arg, "/NODES=", "Nodes per hidden layer (neural networks)", nodes);
		ParseInts(arg, "/LAYERS=", "hidden layer node configurations (deep neural networks)", Layers);
		ParseDouble(arg, "/TOLERANCE=", "Error tolerance", tolerance);
		ParseDouble(arg, "/ALPHA=", "Learning rate Alpha", alpha);
	}

	if (strlen(SaveDirectory) > 0 && save)
	{

		#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

			SaveDir = "./" + std::string(SaveDirectory) + "/";

		#else

			SaveDir = "./" + std::string(SaveDirectory);

		#endif

		fprintf(stderr, "Save Directory: %s\n", SaveDirectory);
	}
	
	if (strlen(InputData) > 0)
	{
		fprintf(stderr, "Input training data: %s\n", InputData);
	}
	
	Layers = Layers.size() > 0 ? Layers : std::vector<int>({ nodes, nodes });
	
	if (RunDNNOptimizer)
		DNNOptimizer(InputData, delimiter, alpha, epochs, tolerance, Layers, save, SaveDirectory, SaveJSON);

	if (RunDNNTrainer)
		DNNTrainer(InputData, delimiter, alpha, epochs, tolerance, Layers, useL2, save, SaveDirectory, SaveJSON);

	if (RunNNTrainer)
		NNTrainer(InputData, delimiter, alpha, nodes, epochs, tolerance, useL2, save, SaveDirectory, SaveJSON);
		
	if (RunNNOptimizer)
		NNOptimizer(InputData, delimiter, alpha, nodes, epochs, tolerance, save, SaveDirectory, SaveJSON);

	return 0;
}
