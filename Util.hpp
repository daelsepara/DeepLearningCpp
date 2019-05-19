#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <cstring>

#include "json.hpp"

#include "MemOps.hpp"
#include "NeuralNetworkOptions.hpp"
#include "NeuralNetwork.hpp"

#include "ConvolutionalNeuralNetworkOptions.hpp"
#include "ConvolutionalNeuralNetwork.hpp"
#include "Layer.hpp"

using json = nlohmann::json;

class Util
{
public:

	static std::vector<std::vector<std::vector<std::vector<double>>>> Convert4DIJ(double* array, int ii, int jj, int xx, int yy)
	{
		std::vector<std::vector<std::vector<std::vector<double>>>> model;
		
		for (auto i = 0; i < ii; i++)
		{
			model.push_back(std::vector<std::vector<std::vector<double>>>());
			
			for (auto j = 0; j < jj; j++)
			{
				model[i].push_back(std::vector<std::vector<double>>());
				
				auto temp = MemOps::New(xx, yy, false);

				MemOps::Copy4DIJ2D(temp, array, xx, yy, jj, ii, i, j);
				
				for (auto y = 0; y < yy; y++)
				{
					model[i][j].push_back(std::vector<double>());
					
					for (auto x = 0; x < xx; x++)
					{
						model[i][j][y].push_back(MemOps::Val2D(temp, xx, yy, x, y));
					}
				}   
				
				MemOps::Free(temp);
			}
		}
		
		return model;
	}
	
	static std::vector<std::vector<double>> Convert2D(double* array, int xx, int yy)
	{
		std::vector<std::vector<double>> model;
		
		for (auto y = 0; y < yy; y++)
		{
			model.push_back(std::vector<double>());
			
			for (auto x = 0; x < xx; x++)
			{
				model[y].push_back(MemOps::Val2D(array, xx, yy, x, y));
			}
		}   
		
		return model;
	}
	
	static std::vector<double> Convert1D(double* array, int ii)
	{
		std::vector<double> model;
		
		for (auto i = 0; i < ii; i++)
		{
			model.push_back(array[i]);
		}
				
		return model;
	}

	static std::vector<double> Vector1D(json j, std::string field, int index)
	{
		auto model = std::vector<double>();
		
		for (int x = 0; x < (int)j[field][index].size(); x++)
		{
			model.push_back(j[field][index][x]);
		}
		
		return model;
	}

	static double* Parse1D(json j, std::string field)
	{
		auto xx = (int)j[field].size();
		
		auto model = MemOps::New(xx, false);
		
		for (int x = 0; x < xx; x++)
		{
			model[x] = j[field][x];
		}
		
		return model;
	}
	
	static double* Parse2D(json j, std::string field)
	{
		auto xx = (int)j[field][0].size();
		auto yy = (int)j[field].size();
		
		auto model = MemOps::New(xx, yy, false);
		
		for (int y = 0; y < yy; y++)
		{
			for (int x = 0; x < xx; x++)
			{
				MemOps::Set2D(model, xx, yy, x, y, j[field][y][x]);
			}
		}
		
		return model;
	}
	
	static double* Parse4DIJ(json json_string, std::string field)
	{
		auto ii = (int)json_string[field].size();
		auto jj = (int)json_string[field][0].size();
		auto yy = (int)json_string[field][0][0].size();
		auto xx = (int)json_string[field][0][0][0].size();
		
		auto model = MemOps::New(xx, yy, 1, ii, jj, false);
		
		for (int i = 0; i < ii; i++)
		{
			for (int j = 0; j < jj; j++)
			{
				auto temp = MemOps::New(xx, yy, false);
				
				for (int y = 0; y < yy; y++)
				{
					for (int x = 0; x < xx; x++)
					{
						MemOps::Set2D(temp, xx, yy, x, y, json_string[field][i][j][y][x]);
					}
				}
				
				MemOps::Copy2D4DIJ(model, temp, ii, jj, xx, yy, i, j);
				
				MemOps::Free(temp);
			}
		}
		
		return model;
	}
	
		static std::string Serialize(NeuralNetwork network, NeuralNetworkOptions opts)
	{
		json j;

		j["Wkj"] = json(Convert2D(network.Wkj, opts.Nodes + 1, opts.Categories));
		j["Wji"] = json(Convert2D(network.Wji, opts.Inputs + 1, opts.Nodes));

		if (network.Min.size() > 0 && network.Max.size() > 0)
		{
			j["Normalization"] += json(network.Min);
			j["Normalization"] += json(network.Max);
		}

		return j.dump();
	}
	
	static std::string Serialize(ConvolutionalNeuralNetwork network, ConvolutionalNeuralNetworkOptions opts)
	{
		json j;
		
		auto InputMaps = 1;
		auto MapSizeX = opts.InputX;
		auto MapSizeY = opts.InputY;
		
		for (auto l = 0; l < (int)network.Layers.size(); l++)
		{
			auto layer = network.Layers[l];
			
			if (layer.Type == LayerTypes::Input)
			{
				j["Layers"] += {{"Type", (int)layer.Type}, {"OutputMaps", layer.OutputMaps}, {"Scale", layer.Scale}, {"KernelSize", layer.KernelSize}};
			}
			else if (layer.Type == LayerTypes::Convolution)
			{
				json FeatureMap = Convert4DIJ(layer.FeatureMap, InputMaps, layer.OutputMaps, layer.KernelSize, layer.KernelSize);
				json Bias = Convert1D(layer.Bias, layer.OutputMaps);
				
				j["Layers"] += {{"Type", (int)layer.Type}, {"OutputMaps", layer.OutputMaps}, {"Scale", layer.Scale}, {"KernelSize", layer.KernelSize}, {"FeatureMap", FeatureMap}, {"Bias", Bias}};
				
				InputMaps = layer.OutputMaps;
				
				MapSizeX -= (layer.KernelSize - 1);
				MapSizeY -= (layer.KernelSize - 1);

			}
			else if (layer.Type == LayerTypes::Subsampling)
			{
				j["Layers"] += {{"Type", (int)layer.Type}, {"OutputMaps", layer.OutputMaps}, {"Scale", layer.Scale}, {"KernelSize", layer.KernelSize}};
				
				MapSizeX /= layer.Scale;
				MapSizeY /= layer.Scale;
			}
		}
		
		int features = MapSizeX * MapSizeY * InputMaps;
		int classes = opts.Classes;
		
		j["Weights"] = json(Convert2D(network.Weights, features, classes));
		j["Bias"] = json(Convert1D(network.Bias, classes));
		
		return j.dump();
	}

	static ConvolutionalNeuralNetwork DeserializeCNN(std::string file_name)
	{
		auto network = ConvolutionalNeuralNetwork();
		
		std::ifstream ifs(file_name);
		
		if (ifs.good())
		{
			json j = json::parse(ifs);
			
			ifs.close();
			
			auto InputMaps = 1;
		
			for (auto i = 0; i < (int)j["Layers"].size(); i++)
			{
				auto layer = j["Layers"][i];
				
				auto Type = static_cast<LayerTypes>((int)layer["Type"]);
				
				if (Type == LayerTypes::Input)
				{
					network.AddLayer(Layer());
				}
				else if (Type == LayerTypes::Convolution)
				{
					auto convolution_layer = Layer((int)layer["OutputMaps"], (int)layer["KernelSize"]);
					
					convolution_layer.FeatureMap = Parse4DIJ(layer, "FeatureMap"); 
					convolution_layer.Bias = Parse1D(layer, "Bias");
					convolution_layer.FeatureMapLength = InputMaps * (int)layer["OutputMaps"] * (int)layer["KernelSize"] * (int)layer["KernelSize"];
					
					InputMaps = (int)layer["OutputMaps"];
					
					network.AddLayer(convolution_layer);
				}
				else if (Type == LayerTypes::Subsampling)
				{
					network.AddLayer(Layer((int)layer["Scale"]));
				}
			}
			
			network.WeightsX = (int)j["Weights"][0].size();
			network.WeightsY = (int)j["Weights"].size();
			network.BiasLength = (int)j["Bias"].size();
			
			network.Bias = Parse1D(j, "Bias");
			network.Weights = Parse2D(j, "Weights");
		}
		
		return network;
	}
	
	static NeuralNetwork DeserializeNN(std::string file_name, NeuralNetworkOptions opts)
	{
		auto network = NeuralNetwork();
		
		std::ifstream ifs(file_name);
		
		if (ifs.good())
		{
			json j = json::parse(ifs);
			
			ifs.close();
			
			network.Wkj = Parse2D(j, "Wkj");
			network.Wji = Parse2D(j, "Wji");
			
			network.Min = Vector1D(j, "Normalization", 0);
			network.Max = Vector1D(j, "Normalization", 1);
		}
		
		return network;
	}
};
#endif
