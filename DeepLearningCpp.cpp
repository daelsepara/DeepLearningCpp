#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ManagedCNN.hpp"
#include "ManagedDNN.hpp"
#include "ManagedNN.hpp"

#include "ManagedUtil.hpp"
#include "Profiler.hpp"
#include "Random.hpp"
#include "ManagedUtil.hpp"

void ParseInt(std::string arg, const char *str, const char *var, int &dst)
{
	auto len = std::string(str).length();

	if (len > 0)
	{
		if (!arg.compare(0, len, str) && arg.length() > len)
		{
			try
			{
				auto val = std::stoi(arg.substr(len));

				std::cerr << "... " << var << " = " << val << std::endl;

				dst = val;
			}
			catch (const std::invalid_argument &ia)
			{
				std::cerr << "... " << var << " = NaN " << ia.what() << std::endl;
				exit(1);
			}
		}
	}
}

void ParseInts(std::string arg, const char *str, const char *var, std::vector<int> &ints)
{
	auto len = std::string(str).length();

	if (len > 0)
	{
		if (!arg.compare(0, len, str) && arg.length() > len)
		{
			try
			{
				auto s = arg.substr(len);

				std::string delimiter = ",";

				size_t pos = 0;

				while ((pos = s.find(delimiter)) != std::string::npos)
				{

					auto val = std::stoi(s.substr(0, pos));

					ints.push_back(val);

					s.erase(0, pos + delimiter.length());
				}

				if (s.length() > 0)
				{
					auto val = std::stoi(s.substr(0, pos));

					ints.push_back(val);
				}
			}
			catch (const std::invalid_argument &ia)
			{
				std::cerr << "... " << var << " = NaN " << ia.what() << std::endl;
				exit(1);
			}
		}
	}
}

void ParseDouble(std::string arg, const char *str, const char *var, double &dst)
{
	auto len = std::string(str).length();

	if (len > 0)
	{
		if (!arg.compare(0, len, str) && arg.length() > len)
		{
			try
			{
				auto val = std::stod(arg.substr(len));

				std::cerr << "... " << var << " = " << std::scientific << val << std::endl;

				dst = val;
			}
			catch (const std::invalid_argument &ia)
			{
				std::cerr << "... " << var << " = NaN " << ia.what() << std::endl;

				exit(1);
			}
		}
	}
}

void Load2D(std::string filename, ManagedArray &input, ManagedArray &output, const char *delimiter, int &inputs, int &categories, int &examples)
{
	auto temp = std::vector<std::vector<double>>();

	std::ifstream file(filename);
	std::string line;

	categories = 0;
	inputs = 0;
	examples = 0;

	while (std::getline(file, line))
	{
		if (line.length() > 0)
		{
			temp.push_back(std::vector<double>());

			std::istringstream is(line);
			std::string token;

			int tokens = 0;

			while (std::getline(is, token, delimiter[0]))
			{
				tokens++;

				auto value = std::stod(token);

				temp[examples].push_back(value);
			}

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

void Load2D(std::string filename, ManagedArray &input, const char *delimiter, int features, int &samples)
{
	auto temp = std::vector<std::vector<double>>();

	std::ifstream file(filename);
	std::string line;

	samples = 0;

	while (std::getline(file, line))
	{
		if (line.length() > 0)
		{
			temp.push_back(std::vector<double>());

			std::istringstream is(line);
			std::string token;

			int tokens = 0;

			while (std::getline(is, token, delimiter[0]))
			{
				tokens++;

				auto value = std::stod(token);

				temp[samples].push_back(value);
			}

			if (tokens > 0 && tokens >= features)
				samples++;
		}
	}

	auto sizey = (int)temp.size();
	auto sizex = (int)temp[0].size();

	input.Resize(features, sizey, false);

	for (auto y = 0; y < sizey; y++)
	{
		for (auto x = 0; x < sizex; x++)
		{
			if (x >= 0 && x < features && (int)temp[y].size() >= features)
				input(x, y) = temp[y][x];
		}
	}

	file.close();
}

void DNNOptimizer(std::string InputData, int delimiter, double alpha, int epochs, double tolerance, std::vector<int> layers, bool save, std::string SaveDirectory, std::string SaveJSON, bool Debug = true)
{
	std::string BaseDirectory = "./";

	if (InputData.length() > 0)
	{
		auto Inputs = 0;
		auto Categories = 0;
		auto Examples = 0;

		auto input = ManagedArray();
		auto output = ManagedArray();

		Load2D(InputData, input, output, delimiter == 0 ? "\t" : ",", Inputs, Categories, Examples);

		std::cerr << std::endl
				  << Examples << " lines read with " << Inputs << " inputs and " << Categories << " categories" << std::endl;

		if (Inputs > 0 && Categories > 0 && Examples > 0 && layers.size() > 0)
		{
			auto opts = NeuralNetworkOptions(alpha, epochs, Categories, Inputs, Examples, tolerance, (int)layers.size(), false);

			auto dnn = ManagedDNN();

			auto start = Profiler::now();

			std::cerr << std::endl
					  << "Optimizing Network..." << std::endl;

			dnn.SetupHiddenLayers(opts.Inputs, opts.Categories, layers);

			auto normalized_input = dnn.Normalize(input);

			dnn.Optimize(normalized_input, output, opts);

			std::cerr << "Optimization Done" << std::endl;

			std::cerr << "elapsed time is " << Profiler::Elapsed(start) << " ms" << std::endl;

			std::cerr << "Final Training error is " << std::scientific << dnn.Cost << std::endl;
			std::cerr << "L2 error is " << std::scientific << dnn.L2 << std::endl;
			std::cerr << "Number of iterations: " << dnn.Iterations << std::endl;

			if (Debug)
			{
				std::cerr << std::endl
						  << "Network Weights:" << std::endl;
				for (auto layer = 0; layer < (int)dnn.Weights.size(); layer++)
				{
					if (layer < (int)dnn.Weights.size() - 1)
					{
						std::cerr << "Layer " << layer << std::endl;
					}
					else
					{
						std::cerr << "Output layer " << layer << std::endl;
					}

					ManagedMatrix::Print2D(dnn.Weights[layer]);
				}
			}

			if (save && SaveJSON.length() > 0)
			{
				std::cerr << std::endl
						  << "Saving Deep Neural Network Parameters" << std::endl;

				ManagedFile::SaveJSON(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, SaveJSON, ManagedUtil::Serialize(dnn));
			}

			dnn.Free();

			ManagedOps::Free(normalized_input);
		}

		ManagedOps::Free(input);
		ManagedOps::Free(output);
	}
}

void DNNTrainer(std::string InputData, int delimiter, double alpha, int epochs, double tolerance, std::vector<int> layers, bool useL2, bool save, std::string SaveDirectory, std::string SaveJSON, bool Debug = true)
{
	std::string BaseDirectory = "./";

	if (InputData.length() > 0)
	{
		auto Inputs = 0;
		auto Categories = 0;
		auto Examples = 0;

		auto input = ManagedArray();
		auto output = ManagedArray();

		Load2D(InputData, input, output, delimiter == 0 ? "\t" : ",", Inputs, Categories, Examples);

		std::cerr << std::endl
				  << Examples << " lines read with " << Inputs << " inputs and " << Categories << " categories" << std::endl;

		if (Inputs > 0 && Categories > 0 && Examples > 0 && layers.size() > 0)
		{
			auto opts = NeuralNetworkOptions(alpha, epochs, Categories, Inputs, Examples, tolerance, (int)layers.size(), useL2);

			auto dnn = ManagedDNN();

			auto start = Profiler::now();

			std::cerr << std::endl
					  << "Training Network..." << std::endl;

			dnn.SetupHiddenLayers(opts.Inputs, opts.Categories, layers);

			auto normalized_input = dnn.Normalize(input);

			dnn.Train(normalized_input, output, opts);

			std::cerr << "Training Done" << std::endl;

			std::cerr << "elapsed time is " << Profiler::Elapsed(start) << " ms" << std::endl;

			std::cerr << "Final Training error is " << std::scientific << dnn.Cost << std::endl;
			std::cerr << "L2 error is " << std::scientific << dnn.L2 << std::endl;
			std::cerr << "Number of iterations: " << dnn.Iterations << std::endl;

			if (Debug)
			{
				std::cerr << std::endl
						  << "Network Weights:" << std::endl;
				for (auto layer = 0; layer < (int)dnn.Weights.size(); layer++)
				{
					if (layer < (int)dnn.Weights.size() - 1)
					{
						std::cerr << "Layer " << layer << std::endl;
					}
					else
					{
						std::cerr << "Output layer " << layer << std::endl;
					}

					ManagedMatrix::Print2D(dnn.Weights[layer]);
				}
			}

			if (save && SaveJSON.length() > 0)
			{
				std::cerr << std::endl
						  << "Saving Deep Neural Network Parameters" << std::endl;

				ManagedFile::SaveJSON(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, SaveJSON, ManagedUtil::Serialize(dnn));
			}

			dnn.Free();

			ManagedOps::Free(normalized_input);
		}

		ManagedOps::Free(input);
		ManagedOps::Free(output);
	}
}

// can also use models trained with DNN Optimizer
void DNNPredict(std::string InputData, std::string ModelFile, int delimiter, int Features, bool save, std::string SaveDirectory, std::string ClassificationFile)
{
	std::string BaseDirectory = "./";

	if (InputData.length() > 0)
	{
		auto Samples = 0;

		auto input = ManagedArray();

		Load2D(InputData, input, delimiter == 0 ? "\t" : ",", Features, Samples);

		std::cerr << std::endl
				  << Samples << " lines read with " << Features << " features" << std::endl;

		if (Features > 0 && Samples > 0)
		{
			auto dnn = ManagedUtil::DeserializeDNN(ModelFile);

			if ((int)dnn.Weights.size() > 0)
			{
				auto normalized = dnn.ApplyNormalization(input);

				std::cerr << std::endl
						  << "Classifying input data..." << std::endl;

				auto start = Profiler::now();

				auto classification = dnn.Classify(normalized, 0.9);

				std::cerr << std::endl
						  << "Classification:" << std::endl;
				ManagedMatrix::PrintList(classification, true);

				std::cerr << std::endl
						  << "Classification Done" << std::endl;
				std::cerr << "elapsed time is " << Profiler::Elapsed(start) << " ms" << std::endl;

				if (save && ClassificationFile.length() > 0)
				{
					ManagedFile::SaveClassification(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, ClassificationFile, classification);
				}

				ManagedOps::Free(classification);
				ManagedOps::Free(normalized);
			}

			dnn.Free();
		}

		ManagedOps::Free(input);
	}
}

void NNTrainer(std::string InputData, int delimiter, double alpha, int Nodes, int epochs, double tolerance, bool useL2, bool save, std::string SaveDirectory, std::string SaveJSON)
{
	std::string BaseDirectory = "./";

	if (InputData.length() > 0)
	{
		auto Inputs = 0;
		auto Categories = 0;
		auto Examples = 0;

		auto input = ManagedArray();
		auto output = ManagedArray();

		Load2D(InputData, input, output, delimiter == 0 ? "\t" : ",", Inputs, Categories, Examples);

		std::cerr << std::endl
				  << Examples << " lines read with " << Inputs << " inputs and " << Categories << " categories" << std::endl;

		if (Inputs > 0 && Categories > 0 && Examples > 0 && Nodes > Inputs)
		{
			auto opts = NeuralNetworkOptions(alpha, epochs, Categories, Inputs, Nodes, Examples, tolerance, useL2);

			auto nn = ManagedNN();

			auto start = Profiler::now();

			std::cerr << std::endl
					  << "Training Network..." << std::endl;

			auto normalized_input = nn.Normalize(input);

			nn.Train(normalized_input, output, opts);

			std::cerr << "Training Done" << std::endl;

			std::cerr << "elapsed time is " << Profiler::Elapsed(start) << " ms" << std::endl;

			std::cerr << "Final Training error is " << std::scientific << nn.Cost << std::endl;
			std::cerr << "L2 error is " << std::scientific << nn.L2 << std::endl;
			std::cerr << "Number of iterations: " << nn.Iterations << std::endl;

			std::cerr << std::endl
					  << "Network Weights:" << std::endl;
			std::cerr << "Weights Wji:" << std::endl;
			ManagedMatrix::Print2D(nn.Wji);
			std::cerr << "Weights Wkj:" << std::endl;
			ManagedMatrix::Print2D(nn.Wkj);

			if (save && SaveJSON.length() > 0)
			{
				std::cerr << std::endl
						  << "Saving Neural Network Parameters" << std::endl;

				ManagedFile::SaveJSON(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, SaveJSON, ManagedUtil::Serialize(nn));
			}

			nn.Free();

			ManagedOps::Free(normalized_input);
		}

		ManagedOps::Free(input);
		ManagedOps::Free(output);
	}
}

void NNOptimizer(std::string InputData, int delimiter, double alpha, int Nodes, int epochs, double tolerance, bool save, std::string SaveDirectory, std::string SaveJSON)
{
	std::string BaseDirectory = "./";

	if (InputData.length() > 0)
	{
		auto Inputs = 0;
		auto Categories = 0;
		auto Examples = 0;

		auto input = ManagedArray();
		auto output = ManagedArray();

		Load2D(InputData, input, output, delimiter == 0 ? "\t" : ",", Inputs, Categories, Examples);

		std::cerr << std::endl
				  << Examples << " lines read with " << Inputs << " inputs and " << Categories << " categories" << std::endl;

		if (Inputs > 0 && Categories > 0 && Examples > 0 && Nodes > Inputs)
		{
			auto opts = NeuralNetworkOptions(alpha, epochs, Categories, Inputs, Nodes, Examples, tolerance);

			auto nn = ManagedNN();

			auto start = Profiler::now();

			std::cerr << std::endl
					  << "Optimizing Network..." << std::endl;

			auto normalized_input = nn.Normalize(input);

			nn.Optimize(normalized_input, output, opts);

			std::cerr << "Optimization Done" << std::endl;

			std::cerr << "elapsed time is " << Profiler::Elapsed(start) << " ms" << std::endl;

			std::cerr << "Final Training error is " << std::scientific << nn.Cost << std::endl;
			std::cerr << "L2 error is " << std::scientific << nn.L2 << std::endl;
			std::cerr << "Number of iterations: " << nn.Iterations << std::endl;

			std::cerr << std::endl
					  << "Network Weights:" << std::endl;
			std::cerr << "Weights Wji:" << std::endl;
			ManagedMatrix::Print2D(nn.Wji);
			std::cerr << "Weights Wkj:" << std::endl;
			ManagedMatrix::Print2D(nn.Wkj);

			if (save && SaveJSON.length() > 0)
			{
				std::cerr << std::endl
						  << "Saving Neural Network Parameters" << std::endl;

				ManagedFile::SaveJSON(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, SaveJSON, ManagedUtil::Serialize(nn));
			}

			nn.Free();

			ManagedOps::Free(normalized_input);
		}

		ManagedOps::Free(input);
		ManagedOps::Free(output);
	}
}

// can also use models trained with NN Optimizer
void NNPredict(std::string InputData, std::string ModelFile, int delimiter, int Features, bool save, std::string SaveDirectory, std::string ClassificationFile)
{
	std::string BaseDirectory = "./";

	if (InputData.length() > 0)
	{
		auto Samples = 0;

		auto input = ManagedArray();

		Load2D(InputData, input, delimiter == 0 ? "\t" : ",", Features, Samples);

		std::cerr << std::endl
				  << Samples << " lines read with " << Features << " features" << std::endl;

		if (Features > 0 && Samples > 0)
		{
			auto nn = ManagedUtil::DeserializeNN(ModelFile);

			if (nn.Wji.Length() > 0 && nn.Wkj.Length() > 0)
			{
				auto normalized = nn.ApplyNormalization(input);

				std::cerr << std::endl
						  << "Classifying input data..." << std::endl;

				auto start = Profiler::now();

				auto classification = nn.Classify(normalized, 0.9);

				std::cerr << std::endl
						  << "Classification:" << std::endl;
				ManagedMatrix::PrintList(classification, true);

				std::cerr << std::endl
						  << "Classification Done" << std::endl;
				std::cerr << "elapsed time is " << Profiler::Elapsed(start) << " ms" << std::endl;

				if (save && ClassificationFile.length() > 0)
				{
					ManagedFile::SaveClassification(SaveDirectory.empty() ? BaseDirectory : SaveDirectory, ClassificationFile, classification);
				}

				ManagedOps::Free(classification);
				ManagedOps::Free(normalized);
			}

			nn.Free();
		}

		ManagedOps::Free(input);
	}
}

void PrintHelp(char **argv)
{
	std::cerr << std::endl
			  << "DeepLearningCpp: Deep Learning Library in C++" << std::endl
			  << std::endl;
	std::cerr << "To use:" << std::endl
			  << std::endl;
	std::cerr << argv[0] << " [options...]" << std::endl
			  << std::endl;

	// Neural network options
	std::cerr << "Network [options]:" << std::endl
			  << std::endl;

	std::cerr << "Single hidden layer network. Used with /NODES option:" << std::endl
			  << std::endl;
	std::cerr << "/NN\t\t\ttrain with single hidden layer neural network using gradient descent backpropagation" << std::endl
			  << std::endl;
	std::cerr << "/NMIN\t\t\ttrain with single hidden layer neural network using an optimizer" << std::endl
			  << std::endl;
	std::cerr << "Deep Neural Network. Used with /LAYERS option:" << std::endl
			  << std::endl;
	std::cerr << "/DNN\t\t\ttrain with a deep neural network" << std::endl
			  << std::endl;
	std::cerr << "/DMIN\t\t\ttrain with a deep neural network but use an optimizer" << std::endl
			  << std::endl;
	std::cerr << "Prediction/Classification:" << std::endl
			  << std::endl;
	std::cerr << "/PREDICT\t\tpredict/classify input data with trained model. Used with /MODEL" << std::endl
			  << std::endl;

	// Network training options
	std::cerr << "Training [options]:" << std::endl
			  << std::endl;
	std::cerr << "/BATCH=[int]\t\tprocess [int] number of datapoints per batch" << std::endl
			  << std::endl;
	std::cerr << "/EPOCH=[int]\t\ttrain neural network for [int] iterations" << std::endl
			  << std::endl;
	std::cerr << "/NODES=[int]\t\tuse [int] neurons in each hidden layer" << std::endl
			  << std::endl;
	std::cerr << "/FEATURES=[int]\t\tspecify [int] number of features per sample datapoint" << std::endl
			  << std::endl;
	std::cerr << "/LAYER=[int1,int2,...]\tcomma-delimited integers that indicate the number layers and neurons per layer" << std::endl
			  << std::endl;
	std::cerr << "/ALPHA=[double]\t\tspecify the learning rate to use while training the network" << std::endl
			  << std::endl;
	std::cerr << "/TOLERANCE=[double]\ttrain network until error falls below [double]" << std::endl
			  << std::endl;
	std::cerr << "/USEL2\t\t\tUse L2 error as tolerance" << std::endl
			  << std::endl;

	// File options
	std::cerr << "File [options]:" << std::endl
			  << std::endl;
	std::cerr << "/INPUT=[string]\t\t2D input data. Each line corresponds to a single datapoint" << std::endl
			  << std::endl;
	std::cerr << "/TAB\t\t\tdata in files are tab-delimited" << std::endl
			  << std::endl;
	std::cerr << "/COMMA\t\t\tdata in files are comma-delimited" << std::endl
			  << std::endl;
	std::cerr << "/SAVE\t\t\tsave trained model. Used with /JSON and/or /SAVEDIR" << std::endl
			  << std::endl;
	std::cerr << "/JSON=[string]\t\tfilename to use when trained model is saved" << std::endl
			  << std::endl;
	std::cerr << "/SAVEDIR=[string]\tsave model/results to a directory" << std::endl
			  << std::endl;
	std::cerr << "/MODEL=[string]\t\tfilename of the trained model to use in /PREDICT" << std::endl
			  << std::endl;
	std::cerr << "/TXT=[string]\t\tsave prediction/classification results to a text file" << std::endl
			  << std::endl;

	// Misc. info
	std::cerr << "Miscellaneous [options]:" << std::endl
			  << std::endl;
	std::cerr << "/HELP or /H\t\tshow available options" << std::endl
			  << std::endl;

	std::cerr << "Powered by: " << std::endl
			  << std::endl;
	std::cerr << "nlohmann/json 3.10.5\thttps://github.com/nlohmann/json" << std::endl
			  << std::endl;
}

int main(int argc, char **argv)
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

	// Prediction
	auto predict = false;
	auto features = 0;

	// common parameters
	auto alpha = 1.0;
	auto epochs = 10000;
	auto useL2 = false; // use L2 error on neural networks or deep neural networks

	// files
	auto save = false;

	char SaveDirectory[200];
	SaveDirectory[0] = '\0';

	std::string SaveDir;

	char SaveJSON[200];
	SaveJSON[0] = '\0';

	char InputData[200];
	InputData[0] = '\0';

	char ModelFile[200];
	ModelFile[0] = '\0';

	char ClassificationFile[200];
	ClassificationFile[0] = '\0';

	int delimiter = 0;

	auto RunDNNOptimizer = false;
	auto RunDNNTrainer = false;
	auto RunNNOptimizer = false;
	auto RunNNTrainer = false;

	auto Debug = true;

	if (argc > 1)
	{
		bool help = false;

		if (argc == 2)
		{
			std::string arg = argv[1];
			std::transform(arg.begin(), arg.end(), arg.begin(), ::toupper);

			help = (!arg.compare("/HELP") || !arg.compare("/H"));
		}

		if (help)
		{
			PrintHelp(argv);
		}
		else
		{
			for (auto i = 1; i < argc; i++)
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
				else if (!arg.compare("/PREDICT"))
				{
					predict = true;
				}
				else if (!arg.compare("/DEBUG"))
				{
					Debug = true;
				}
				else if (!arg.compare("/NODEBUG"))
				{
					Debug = false;
				}

				if (!arg.compare(0, 9, "/SAVEDIR=") && arg.length() > 9)
				{
					std::copy(&argv[i][9], &argv[i][9] + sizeof(SaveDirectory), SaveDirectory);
				}

				if (!arg.compare(0, 6, "/JSON=") && arg.length() > 6)
				{
					std::copy(&argv[i][6], &argv[i][6] + sizeof(SaveJSON), SaveJSON);
				}

				if (!arg.compare(0, 7, "/INPUT=") && arg.length() > 7)
				{
					std::copy(&argv[i][7], &argv[i][7] + sizeof(InputData), InputData);
				}

				if (!arg.compare(0, 7, "/MODEL=") && arg.length() > 7)
				{
					std::copy(&argv[i][7], &argv[i][7] + sizeof(ModelFile), ModelFile);
				}

				if (!arg.compare(0, 5, "/TXT=") && arg.length() > 5)
				{
					std::copy(&argv[i][5], &argv[i][5] + sizeof(ClassificationFile), ClassificationFile);
				}

				ParseInt(arg, "/BATCH=", "Batch Size", batchsize);
				ParseInt(arg, "/EPOCH=", "Epochs", epochs);
				ParseInt(arg, "/NODES=", "Nodes per hidden layer (neural networks)", nodes);
				ParseInt(arg, "/FEATURES=", "# features per data point", features);
				ParseInts(arg, "/LAYERS=", "hidden layer node configurations (deep neural networks)", Layers);
				ParseDouble(arg, "/TOLERANCE=", "Error tolerance", tolerance);
				ParseDouble(arg, "/ALPHA=", "Learning rate Alpha", alpha);
			}

			if (std::string(SaveDirectory).length() > 0 && save)
			{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)

				SaveDir = "./" + std::string(SaveDirectory) + "/";

#else

				SaveDir = "./" + std::string(SaveDirectory);

#endif

				std::cerr << "... Save Directory: " << SaveDir << std::endl;
			}

			if (std::string(InputData).length() > 0)
			{
				std::cerr << "... Input training data: " << InputData << std::endl;
			}

			if (std::string(ModelFile).length() > 0)
			{
				std::cerr << "... Model File: " << ModelFile << std::endl;
			}

			if (std::string(SaveJSON).length() > 0)
			{
				std::cerr << "... JSON File: " << SaveJSON << ".json" << std::endl;
			}

			if (std::string(ClassificationFile).length() > 0)
			{
				std::cerr << "... Classification File: " << ClassificationFile << ".txt" << std::endl;
			}

			Layers = Layers.size() > 0 ? Layers : std::vector<int>({nodes, nodes});

			if (RunDNNOptimizer)
			{
				if (predict)
				{
					DNNPredict(InputData, ModelFile, delimiter, features, save, SaveDir, ClassificationFile);
				}
				else
				{
					DNNOptimizer(InputData, delimiter, alpha, epochs, tolerance, Layers, save, SaveDir, SaveJSON, Debug);
				}
			}

			if (RunDNNTrainer)
			{
				if (predict)
				{
					DNNPredict(InputData, ModelFile, delimiter, features, save, SaveDir, ClassificationFile);
				}
				else
				{
					DNNTrainer(InputData, delimiter, alpha, epochs, tolerance, Layers, useL2, save, SaveDir, SaveJSON, Debug);
				}
			}

			if (RunNNTrainer)
			{
				if (predict)
				{
					NNPredict(InputData, ModelFile, delimiter, features, save, SaveDir, ClassificationFile);
				}
				else
				{
					NNTrainer(InputData, delimiter, alpha, nodes, epochs, tolerance, useL2, save, SaveDir, SaveJSON);
				}
			}

			if (RunNNOptimizer)
			{
				if (predict)
				{
					NNPredict(InputData, ModelFile, delimiter, features, save, SaveDir, ClassificationFile);
				}
				else
				{
					NNOptimizer(InputData, delimiter, alpha, nodes, epochs, tolerance, save, SaveDir, SaveJSON);
				}
			}
		}
	}
	else
	{
		PrintHelp(argv);
	}

	return 0;
}
