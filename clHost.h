// BEAM OpenCL Miner
// OpenCL Host Interface
// Copyright 2018 The Beam Team	
// Copyright 2018 Wilke Trei
// Copyright 2019 Andrei Dimitrief-Jianu

#include <CL/cl.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <cstdlib>
#include <climits>

#include "beamStratum.h"

namespace beamMiner 
{

struct clCallbackData 
{
	uint32_t gpuIndex;
	beamStratum* stratum;
	beamStratum::WorkDescription workDescription;
	void* clHost;
};

class clHost 
{
	private:
	// OpenCL 
	vector<cl::Platform> platforms;  
	vector<cl::Context> contexts;
	vector<cl::CommandQueue> queues;
	vector<cl::Device> devices;
	vector<cl::Event> events;
	vector<unsigned*> results;

	vector< vector<cl::Buffer> > buffers;
	vector< vector<cl::Kernel> > kernels;

	vector<bool> is3G;

	// Statistics
	vector<int> solutionCnt;

	// To check if a mining thread stoped and we must resume it
	vector<bool> paused;

	vector<int32_t> intensities;

	// Callback data
	vector<clCallbackData> currentWork;
	bool restart = true;

	// Functions
	void detectPlatformDevices(vector<int32_t>, vector<int32_t>, bool, bool);
	bool loadAndCompileKernel(cl::Device &, uint32_t, bool);
	void queueKernels(uint32_t, clCallbackData*);
	void queueWork(uint32_t, clCallbackData*); 
	
	// The connectors
	beamStratum* minerStratum;

	atomic_uint64_t workCounter;
	uint64_t workCounterMinModulo;
	uint64_t workCounterMaxModulo;

	public:
	
	clHost(beamStratum*, vector<int32_t>, vector<int32_t>, bool, bool);
	void startMining();	
	void callbackFunc(cl_int, void*);
};

}
