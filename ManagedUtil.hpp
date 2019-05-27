#ifndef MANAGED_UTIL_HPP
#define MANAGED_UTIL_HPP

#include <vector>
#include <cstring>

#include "json.hpp"

#include "ManagedArray.hpp"

#include "NeuralNetworkOptions.hpp"
#include "ManagedDNN.hpp"
#include "ManagedNN.hpp"
#include "OptimizerDNN.hpp"
#include "OptimizerNN.hpp"

#include "ConvolutionalNeuralNetworkOptions.hpp"
#include "ManagedCNN.hpp"
#include "ManagedLayer.hpp"

using json = nlohmann::json;

class ManagedUtil
{
public:

	static std::vector<std::vector<double>> Convert2D(ManagedArray array)
	{
		std::vector<std::vector<double>> model;

		if (array.Length() > 0)
		{
			for (auto y = 0; y < array.y; y++)
			{
				model.push_back(std::vector<double>());

				for (auto x = 0; x < array.x; x++)
					model[y].push_back(array(x, y));
			}
		}

		return model;
	}

	static std::vector<double> Convert1D(ManagedArray array)
	{
		std::vector<double> model;
		
		for (auto i = 0; i < array.Length(); i++)
		{
			model.push_back(array(i));
		}
				
		return model;
	}
	
	static std::vector<std::vector<std::vector<std::vector<double>>>> Convert4DIJ(ManagedArray array)
	{
		std::vector<std::vector<std::vector<std::vector<double>>>> model;
		
		for (auto i = 0; i < array.i; i++)
		{
			model.push_back(std::vector<std::vector<std::vector<double>>>());
			
			for (auto j = 0; j < array.j; j++)
			{
				model[i].push_back(std::vector<std::vector<double>>());
				
				auto temp = ManagedArray(array.x, array.y);

				ManagedOps::Copy4DIJ2D(temp, array, i, j);
				
				for (auto y = 0; y < temp.y; y++)
				{
					model[i][j].push_back(std::vector<double>());
					
					for (auto x = 0; x < temp.x; x++)
					{
						model[i][j][y].push_back(temp(x, y));
					}
				}   
				
				ManagedOps::Free(temp);
			}
		}
		
		return model;
	}
	
	static ManagedArray Parse1D(json j, std::string field)
	{
		auto model = ManagedArray((int)j[field].size());
		
		for (int x = 0; x < (int)j[field].size(); x++)
		{
			model(x) = j[field][x];
		}
		
		return model;
	}
	
	static ManagedArray Parse2D(json j, std::string field)
	{
		auto model = ManagedArray((int)j[field][0].size(), (int)j[field].size());
		
		for (int y = 0; y < (int)j[field].size(); y++)
		{
			for (int x = 0; x < (int)j[field][0].size(); x++)
			{
				model(x, y) = j[field][y][x];
			}
		}
		
		return model;
	}
	
	static ManagedArray Parse2D(json j, std::string field, int index)
	{
		auto model = ManagedArray((int)j[field][index][0].size(), (int)j[field][index].size());
		
		for (int y = 0; y < (int)j[field][index].size(); y++)
		{
			for (int x = 0; x < (int)j[field][index][0].size(); x++)
			{
				model(x, y) = j[field][index][y][x];
			}
		}
		
		return model;
	}
	
	static ManagedArray Parse4DIJ(json json_string, std::string field)
	{
		auto ii = (int)json_string[field].size();
		auto jj = (int)json_string[field][0].size();
		auto yy = (int)json_string[field][0][0].size();
		auto xx = (int)json_string[field][0][0][0].size();
		
		auto model = ManagedArray(xx, yy, 1, ii, jj);
		
		auto temp = ManagedArray(xx, yy);
		
		for (int i = 0; i < ii; i++)
		{
			for (int j = 0; j < jj; j++)
			{
				for (auto y = 0; y < yy; y++)
				{
					for (auto x = 0; x < xx; x++)
					{
						temp(x, y) = json_string[field][i][j][y][x];
					}
				}   
				
				ManagedOps::Copy2D4DIJ(model, temp, i, j);
			}
		}
		
		ManagedOps::Free(temp);
		
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

	static std::string Serialize(OptimizerDNN network)
	{
		json j;

		for (auto layer = 0; layer < (int)network.Weights.size(); layer++)
		{
			j["Weights"] += json(Convert2D(network.Weights[layer]));
		}

		if (network.Min.size() > 0 && network.Max.size() > 0)
		{
			j["Normalization"] += json(network.Min);
			j["Normalization"] += json(network.Max);
		}

		return j.dump();
	}

	static std::string Serialize(ManagedDNN network)
	{
		json j;

		for (auto layer = 0; layer < (int)network.Weights.size(); layer++)
		{
			j["Weights"] += json(Convert2D(network.Weights[layer]));
		}

		if (network.Min.size() > 0 && network.Max.size() > 0)
		{
			j["Normalization"] += json(network.Min);
			j["Normalization"] += json(network.Max);
		}

		return j.dump();
	}

	static std::string Serialize(ManagedNN network)
	{
		json j;

		j["Wkj"] = json(Convert2D(network.Wkj));
		j["Wji"] = json(Convert2D(network.Wji));

		if (network.Min.size() > 0 && network.Max.size() > 0)
		{
			j["Normalization"] += json(network.Min);
			j["Normalization"] += json(network.Max);
		}

		return j.dump();
	}

	static std::string Serialize(OptimizerNN network)
	{
		json j;

		j["Wkj"] = json(Convert2D(network.Wkj));
		j["Wji"] = json(Convert2D(network.Wji));

		if (network.Min.size() > 0 && network.Max.size() > 0)
		{
			j["Normalization"] += json(network.Min);
			j["Normalization"] += json(network.Max);
		}

		return j.dump();
	}

	static std::string Serialize(ManagedCNN network)
	{
		json j;
		
		for (auto l = 0; l < (int)network.Layers.size(); l++)
		{
			auto layer = network.Layers[l];
			
			if (layer.Type == LayerTypes::Input)
			{
				j["Layers"] += {{"Type", (int)layer.Type}, {"OutputMaps", layer.OutputMaps}, {"Scale", layer.Scale}, {"KernelSize", layer.KernelSize}};
			}
			else if (layer.Type == LayerTypes::Convolution)
			{
				json FeatureMap = Convert4DIJ(layer.FeatureMap);
				json Bias = Convert1D(layer.Bias);
				
				j["Layers"] += {{"Type", (int)layer.Type}, {"OutputMaps", layer.OutputMaps}, {"Scale", layer.Scale}, {"KernelSize", layer.KernelSize}, {"FeatureMap", FeatureMap}, {"Bias", Bias}};
			}
			else if (layer.Type == LayerTypes::Subsampling)
			{
				j["Layers"] += {{"Type", (int)layer.Type}, {"OutputMaps", layer.OutputMaps}, {"Scale", layer.Scale}, {"KernelSize", layer.KernelSize}};
			}
		}
		
		j["Weights"] = json(Convert2D(network.Weights));
		j["Bias"] = json(Convert1D(network.Bias));
		
		return j.dump();
	}
	
	static ManagedNN DeserializeNN(std::string file_name)
	{
		auto network = ManagedNN();
		
		std::ifstream ifs(file_name);
		
		if (ifs.good())
		{
			json j = json::parse(ifs);
			
			ifs.close();
			
			network.Wji = Parse2D(j, "Wji");
			network.Wkj = Parse2D(j, "Wkj");
			
			network.Min = Vector1D(j, "Normalization", 0);
			network.Max = Vector1D(j, "Normalization", 1);
		}
		
		return network;
	}
	
	static ManagedDNN DeserializeDNN(std::string file_name)
	{
		auto network = ManagedDNN();
		
		std::ifstream ifs(file_name);
		
		if (ifs.good())
		{
			json j = json::parse(ifs);
			
			ifs.close();
			
			auto layers = (int)j["Weights"].size();
			
			network.Weights.clear();
			network.X.clear();
			network.Z.clear();
			network.Activations.clear();

			for (auto layer = 0; layer < layers; layer++)
			{
				network.Weights.push_back(Parse2D(j, "Weights", layer));
				network.X.push_back(ManagedArray());
				network.Z.push_back(ManagedArray());
				network.Deltas.push_back(ManagedArray());
				network.D.push_back(ManagedArray());

				if (layer < layers - 1)
					network.Activations.push_back(ManagedArray());
			}
			
			network.Min = Vector1D(j, "Normalization", 0);
			network.Max = Vector1D(j, "Normalization", 1);
		}
		
		return network;
	}
	
	static ManagedCNN DeserializeCNN(std::string file_name)
	{
		auto network = ManagedCNN();
		
		std::ifstream ifs(file_name);
		
		if (ifs.good())
		{
			json j = json::parse(ifs);
			
			ifs.close();
			
			for (auto i = 0; i < (int)j["Layers"].size(); i++)
			{
				auto layer = j["Layers"][i];
				
				auto Type = static_cast<LayerTypes>((int)layer["Type"]);
				
				if (Type == LayerTypes::Input)
				{
					network.AddLayer(ManagedLayer());
				}
				else if (Type == LayerTypes::Convolution)
				{
					auto convolution_layer = ManagedLayer((int)layer["OutputMaps"], (int)layer["KernelSize"]);
					
					convolution_layer.FeatureMap = Parse4DIJ(layer, "FeatureMap");
					convolution_layer.Bias = Parse1D(layer, "Bias");
					
					network.AddLayer(convolution_layer);
				}
				else if (Type == LayerTypes::Subsampling)
				{
					network.AddLayer(ManagedLayer((int)layer["Scale"]));
				}
			}
			
			network.Bias = Parse1D(j, "Bias");
			network.Weights = Parse2D(j, "Weights");
		}
		
		return network;
	}
};
#endif
