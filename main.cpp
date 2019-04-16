// BEAM OpenCL Miner
// Main Function
// Copyright 2018 The Beam Team	
// Copyright 2018 Wilke Trei
// Copyright 2019 Andrei Dimitrief-Jianu

#include "beamStratum.h"
#include "clHost.h"
#include "base64.h"

#define VERSION "1.1.0"

inline vector<string> &split(const string &s, char delim, vector<string> &elems) 
{
    stringstream ss(s);
    string item;
    while(getline(ss, item, delim)) 
	{
        elems.push_back(item);
    }
    
	return elems;
}

inline vector<string> split(const string &s, char delim) 
{
    vector<string> elems;

    return split(s, delim, elems);
}

inline std::string encrypt(std::string plain, std::string const& key)
{
	if (key.size())
	{
		for (std::size_t i = 0; i < plain.size(); i++)
		{
			plain[i] ^= key[i%key.size()];
		}
	}

	return plain;
}

uint32_t cmdParser(
	vector<string> args, 
	vector<string> &hosts, 
	vector<string> &ports, 
	vector<string> &minerCredentials, 
	vector<int32_t> &devices, 
	vector<int32_t> &intensities, 
	bool &debug, 
	bool &cpuMine, 
	bool &force3G ) 
{
	// exit if empy command line
	if (args.size() < 2)
	{
		return 0x8;
	}

	bool hostSet = false;
	bool invalidIntensityValue = false;
	
	for (size_t i = 1; i < args.size(); i++) 
	{
		if ((args[i].compare("-h")  == 0) || (args[i].compare("--help")  == 0)) 
		{
			return 0x8;
		}

		if (args[i].compare("--server")  == 0) 
		{
			if (i+1 < args.size()) 
			{
				vector<string> tmp = split(args[i+1], ':');
				if (tmp.size() == 3) 
				{
					hosts.push_back(tmp[0]);
					ports.push_back(tmp[1]);
					minerCredentials.push_back(tmp[2]);
					hostSet = true;	
					i++;
					continue;
				}
			}
			else
			{
				return 0x8;
			}
		}

		if (args[i].compare("--devices")  == 0) 
		{
			if (i+1 < args.size()) 
			{
				vector<string> tmp = split(args[i+1], ',');
				for (int j = 0; j < tmp.size(); j++) 
				{
					devices.push_back(stoi(tmp[j]));
				}
				i++;
				continue;
			}
			else
			{
				return 0x8;
			}
		}

		if (args[i].compare("--intensity") == 0) 
		{
			if (i+1 < args.size()) 
			{
				vector<string> tmp = split(args[i+1], ',');
				for (int j = 0; j < tmp.size(); j++)
				{
					int32_t intensity = stoi(tmp[j]);
					if (intensity < 0 || 999 < intensity) invalidIntensityValue = true;
					intensities.push_back(intensity);
				}
				i++;
				continue;
			}
			else
			{
				return 0x8;
			}
		}

		if (args[i].compare("--force3G")  == 0) 
		{
			force3G = true;
			continue;
		}

		if (args[i].compare("--enable-cpu")  == 0) 
		{
			cpuMine = true;
			continue;
		}

		if (args[i].compare("--debug")  == 0) 
		{
			debug = true;
			continue;
		}
	
		if (args[i].compare("--version") == 0) 
		{
			cout << VERSION << endl;
			exit(0);
		}

		cout << "unknown parameter: " << args[i] << endl;
		exit(0);
	}

	uint32_t result = 0;

	if (!hostSet) result += 1;
	
	if (invalidIntensityValue)
	{
		result += 4;
	}
	else if (0 == devices.size() && 0 == intensities.size())
	{
		intensities.push_back(999);
	}
	else if (0 == devices.size() && 1 == intensities.size())
	{
		// this is OK... one intensity value for all devices;
	}
	else if (devices.size() != intensities.size())
	{
		result += 4;
	}

	return result;
}

int main(int argc, char* argv[]) 
{
	vector<string> cmdLineArgs(argv, argv+argc);
	vector<string> hosts;
	vector<string> ports;
	vector<string> minerCredentials;
	bool debug = false;
	bool cpuMine = false;
	bool useTLS = true;
	vector<int32_t> devices;
	vector<int32_t> intensities;
	bool force3G = false;

	vector<beamMiner::clHost*> clHosts;
	vector<beamMiner::beamStratum*> minerStratums;

	uint32_t parsed = cmdParser(cmdLineArgs, hosts, ports, minerCredentials, devices, intensities, debug, cpuMine, force3G);

	cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
	cout << "   BEAM OpenCL miner         " << endl;
	cout << "   v" << VERSION << endl;
	cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;

	if (0 != parsed) 
	{
		cout << endl;

		if (parsed & 0x1) 
		{
			cout << "Error: Parameter --server missing" << endl;
		}

		if (parsed & 0x4)
		{
			cout << "Error: Parameter --intensity invalid value" << endl;
		}
		
		cout << endl;
		cout << "Parameters: " << endl;
		cout << " --help / -h " << "\t\t\t\tShow this message" << endl;
		cout << " --server <server>:<port>:<key> " << "\tThe BEAM stratum server, port, and API key (required)" << endl;
		cout << " --devices <numbers> " << "\t\t\tA comma-separated list of devices that should be used for mining (default: all)" << endl; 
		cout << " --intensity <intensity> " << "\t\tThe miner intensity(ies) (if more than one, comma-separated; takes values from 0 to 999; default: 999)" << endl;
		cout << " --enable-cpu " << "\t\t\t\tEnable mining on OpenCL CPU devices" << endl;
		cout << " --force3G	" << "\t\t\tForce miner to use max 3GB for all installed GPUs" << endl;
		cout << " --debug " << "\t\t\t\tPrint debugging info" << endl;
		cout << " --version	" << "\t\t\tPrint the version number" << endl;
		cout << endl;

		exit(0);
	}

	cout << endl;
	cout << "Parameters:" << endl;
	cout << ">>>>>>>>>>>" << endl;
	for (size_t i = 0; i < hosts.size(); i++)
	{
		cout << "Server:    " << hosts[i] << ":" << ports[i] << ":" << minerCredentials[i] << endl;
	}
	if (devices.empty())
	{
		cout << "Intensity: " << intensities[0] << endl;
	}
	else
	{
		for (size_t i = 0; i < devices.size(); i++)
		{
			int32_t intensity = (1 == intensities.size())?(intensities[0]):(intensities[i]);
			cout << "Device|Intensity: " << devices[i] << "|" << intensity << endl;
		}
	}
	if (cpuMine)
	{
		cout << "CPU mining enabled" << endl;
	}
	if (force3G)
	{
		cout << "GPU kernels forced to 3GB" << endl;
	}

	for (size_t i = 0; i < hosts.size(); i++)
	{
		beamMiner::beamStratum *minerStratum = new beamMiner::beamStratum(hosts[i], ports[i], minerCredentials[i], debug, false);

		cout << endl;
		cout << "Host: " << hosts[i] << ":" << ports[i] << endl;
		cout << "Setup OpenCL devices:" << endl;
		cout << ">>>>>>>>>>>>>>>>>>>>>" << endl;
		
		beamMiner::clHost *clHost = new beamMiner::clHost(minerStratum, devices, intensities, cpuMine, force3G);

		minerStratums.push_back(minerStratum);
		clHosts.push_back(clHost);
	}
	
	size_t hostIndex = 0;
	while (true)
	{
		clHosts[hostIndex]->startMining();
		hostIndex = ++hostIndex % clHosts.size();
		
		cout << endl << "Switching host..." << endl;

		this_thread::sleep_for(std::chrono::milliseconds(200));
	}

}

#if defined(_MSC_VER) && (_MSC_VER >= 1900)

FILE _iob[] = { *stdin, *stdout, *stderr };
extern "C" FILE * __cdecl __iob_func(void) { return _iob; }

#endif
