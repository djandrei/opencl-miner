// BEAM OpenCL Miner
// OpenCL Host Interface
// Copyright 2018 The Beam Team	
// Copyright 2018 Wilke Trei
// Copyright 2019 Andrei Dimitrief-Jianu

#include "clHost.h"
#include "./kernels/equihash_150_5_inc.h"

namespace beamMiner 
{

// Helper function for string manipulation
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

// Helper function that tests if a OpenCL device supports a certain CL extension
inline bool hasExtension(cl::Device &device, string extension) 
{
	string info;
	device.getInfo(CL_DEVICE_EXTENSIONS, &info);
	vector<string> extens = split(info, ' ');

	for (int i=0; i<extens.size(); i++) 
	{
		if (extens[i].compare(extension) == 0) 	return true;
	} 

	return false;
}

// This is a bit ugly c-style, but the OpenCL headers are initially for c and
// support c-style callback functions (no member functions) only.
// This function will be called every time a GPU is done with its current work
void CL_CALLBACK CCallbackFunc(cl_event ev, cl_int err , void* data) 
{
	clHost* self = static_cast<clHost*>(((clCallbackData*) data)->clHost);
	self->callbackFunc(err, data);
}

// Function to load the OpenCL kernel and prepare our device for mining
bool clHost::loadAndCompileKernel(cl::Device &device, uint32_t pl, bool use3G) 
{
	cout << "   Loading and compiling Beam OpenCL Kernel" << endl;

	// reading the kernel
	string progStr = string(__equihash_150_5_cl, __equihash_150_5_cl_len); 

	cl::Program::Sources source(1,std::make_pair(progStr.c_str(), progStr.length()+1));

	// Create a program object and build it
	vector<cl::Device> devicesTMP;
	devicesTMP.push_back(device);

	cl::Program program(contexts[pl], source);
	cl_int err;
	if (!use3G) 
	{
		err = program.build(devicesTMP,"");
	}
	else 
	{
		err = program.build(devicesTMP,"-DMEM3G");
	}

	// Check if the build was Ok
	if (!err) 
	{
		cout << "   Build sucessfull. " << endl;

		// Store the device and create a queue for it
		cl_command_queue_properties queue_prop = 0;  
		devices.push_back(device);
		queues.push_back(cl::CommandQueue(contexts[pl], devices[devices.size()-1], queue_prop, NULL)); 

		// Reserve events, space for storing results and so on
		events.push_back(cl::Event());
		results.push_back(NULL);
		currentWork.push_back(clCallbackData());
		paused.push_back(true);
		is3G.push_back(use3G);
		solutionCnt.push_back(0);

		// Create the kernels
		vector<cl::Kernel> newKernels;	
		newKernels.push_back(cl::Kernel(program, "clearCounter", &err));
		newKernels.push_back(cl::Kernel(program, "round0", &err));
		newKernels.push_back(cl::Kernel(program, "round1", &err));
		newKernels.push_back(cl::Kernel(program, "round2", &err));
		newKernels.push_back(cl::Kernel(program, "round3", &err));
		newKernels.push_back(cl::Kernel(program, "round4", &err));
		newKernels.push_back(cl::Kernel(program, "round5", &err));
		if (use3G) 
		{
			newKernels.push_back(cl::Kernel(program, "combine3G", &err));
			newKernels.push_back(cl::Kernel(program, "repack", &err));
			newKernels.push_back(cl::Kernel(program, "move", &err));
		} 
		else 
		{
			newKernels.push_back(cl::Kernel(program, "combine", &err));
		}
		kernels.push_back(newKernels);

		// Create the buffers
		vector<cl::Buffer> newBuffers;
		
		if (!use3G) 
		{
			newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint4) * 71303168, NULL, &err));
			newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint4) * 71303168, NULL, &err)); 
			newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint4) * 71303168, NULL, &err)); 
			newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint2) * 71303168, NULL, &err)); 
		} 
		else 
		{
			newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint4) * 69599232, NULL, &err));
			newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint4) * 69599232, NULL, &err)); 
			newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint4) * 52199424, NULL, &err)); 
			newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint2) * 1, NULL, &err)); 
		}

		newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint4) * 256, NULL, &err));   
		newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint) * 49152, NULL, &err));  
		newBuffers.push_back(cl::Buffer(contexts[pl], CL_MEM_READ_WRITE,  sizeof(cl_uint) * 324, NULL, &err));  
		buffers.push_back(newBuffers);		
		
		return true;
	} 
	else 
	{
		cout << "   Program build error, device will not be used. " << endl;
		// Print error msg so we can debug the kernel source
		cout << "   Build Log: "     << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devicesTMP[0]) << endl;

		return false;
	}
}

// Detect the OpenCL hardware on this system
void clHost::detectPlatformDevices(vector<int32_t> selectedDevices, vector<int32_t> selectedIntensities, bool allowCPU, bool force3G) 
{
	// read the OpenCL platforms on this system
	cl::Platform::get(&platforms);  

	// this is for enumerating the devices
	uint32_t currentDevice = 0;
	
	for (size_t pl = 0; pl < platforms.size(); pl++) 
	{
		// Create the OpenCL contexts, one for each platform
		cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[pl](), 0};  
		cl::Context context;
		if (allowCPU) 
		{ 
			context = cl::Context(CL_DEVICE_TYPE_ALL, properties);
		} 
		else 
		{
			context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
		}
		contexts.push_back(context);

		// Read the devices of this platform
		vector< cl::Device > nDev = context.getInfo<CL_CONTEXT_DEVICES>();  
		for (uint32_t di=0; di<nDev.size(); di++) 
		{	
			// Print the device name
			string name;
			if ( hasExtension(nDev[di], "cl_amd_device_attribute_query") ) 
			{
				// on AMD this gives more readable result
				nDev[di].getInfo(0x4038,&name);
			} 
			else 
			{
				// on AMD this gives more readable result
				nDev[di].getInfo(CL_DEVICE_NAME, &name);
			}

			// Get rid of strange characters at the end of device name
			if (isalnum((int) name.back()) == 0) 
			{
				name.pop_back();
			} 
			
			cout << "Found device " << currentDevice << ": " << name << endl;

			// Check if the device should be selected
			bool pickDevice = false;
			int32_t intensity;
			if (0 == selectedDevices.size())
			{
				pickDevice = true;
				intensity = selectedIntensities[0];
			} 
			else if (selectedDevices.end() != find(selectedDevices.begin(), selectedDevices.end(), currentDevice))
			{
				pickDevice = true;
				intensity = selectedIntensities[distance(selectedDevices.begin(), find(selectedDevices.begin(), selectedDevices.end(), currentDevice))];
			}

			if (pickDevice) 
			{
				// Check if the CPU / GPU has enough memory
				uint64_t deviceMemory = nDev[di].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
				uint64_t needed_4G = 7* ((uint64_t) 570425344) + 4096 + 196608 + 1296;
				uint64_t needed_3G = 4* ((uint64_t) 556793856) + ((uint64_t) 835190784) + 4096 + 196608 + 1296;

				cout << "   Device reports " << deviceMemory / (1024*1024) << "MByte total memory" << endl;

				if ( hasExtension(nDev[di], "cl_amd_device_attribute_query") ) 
				{
					uint64_t freeDeviceMemory;
				 	nDev[di].getInfo(0x4039, &freeDeviceMemory);  // CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
					freeDeviceMemory *= 1024;
					cout << "   Device reports " << freeDeviceMemory / (1024*1024) << "MByte free memory (AMD)" << endl;
					deviceMemory = min<uint64_t>(deviceMemory, freeDeviceMemory);
				}				

				bool loadedKernel = false;
				if ((deviceMemory > needed_4G) && (!force3G)) 
				{ 
					cout << "   Memory check for 4G kernel passed" << endl;
					loadedKernel = loadAndCompileKernel(nDev[di], pl, false);
				} 
				else if (deviceMemory > needed_3G) 
				{
					cout << "   Memory check for 3G kernel passed" << endl;
					loadedKernel = loadAndCompileKernel(nDev[di], pl, true);
				}  
				else 
				{
					cout << "   Memory check failed, required minimum memory: " << needed_3G/(1024*1024) << endl;
				}

				if (loadedKernel) intensities.push_back(intensity);
			} 
			else 
			{
				cout << "   Device will not be used, it was not included in --devices parameter." << endl;
			}

			currentDevice++; 
		}
	}

	if (devices.size() == 0) 
	{
		cout << "No compatible OpenCL devices found or all are deselected. Exiting..." << endl;
		exit(0);
	}
}

// Setup function called from outside
clHost::clHost(
	beamStratum* minerStratumIn, 
	vector<int32_t> selectedDevices, 
	vector<int32_t> selectedIntensities, 
	bool allowCPU, 
	bool force3G) 
{
	workCounterMinModulo = 960;
	workCounterMaxModulo = 1000;

	srand(time(0));
	workCounter = rand() % workCounterMaxModulo; // generate random start value from 1 to max modulo

	minerStratum = minerStratumIn;
	
	detectPlatformDevices(selectedDevices, selectedIntensities, allowCPU, force3G);
}

// Function that will catch new work from the stratum interface and then queue the work on the device
void clHost::queueKernels(uint32_t gpuIndex, clCallbackData* workData) 
{
	cl_ulong4 work;	
	cl_ulong nonce;

	// Get a new set of work from the stratum interface
	workData->stratum->getWork(workData->workDescription, (uint8_t *) &work);
	nonce = workData->workDescription.nonce;

	if (!is3G[gpuIndex]) 
	{		
		// Starting the 4G kernels
		
		// Kernel arguments for cleanCounter
		kernels[gpuIndex][0].setArg(0, buffers[gpuIndex][5]); 
		kernels[gpuIndex][0].setArg(1, buffers[gpuIndex][6]);

		// Kernel arguments for round0
		kernels[gpuIndex][1].setArg(0, buffers[gpuIndex][0]); 
		kernels[gpuIndex][1].setArg(1, buffers[gpuIndex][2]); 
		kernels[gpuIndex][1].setArg(2, buffers[gpuIndex][5]); 
		kernels[gpuIndex][1].setArg(3, work); 
		kernels[gpuIndex][1].setArg(4, nonce); 

		// Kernel arguments for round1
		kernels[gpuIndex][2].setArg(0, buffers[gpuIndex][0]); 
		kernels[gpuIndex][2].setArg(1, buffers[gpuIndex][2]); 
		kernels[gpuIndex][2].setArg(2, buffers[gpuIndex][1]); 
		kernels[gpuIndex][2].setArg(3, buffers[gpuIndex][3]); 	// Index tree will be stored here
		kernels[gpuIndex][2].setArg(4, buffers[gpuIndex][5]); 

		// Kernel arguments for round2
		kernels[gpuIndex][3].setArg(0, buffers[gpuIndex][1]); 
		kernels[gpuIndex][3].setArg(1, buffers[gpuIndex][0]);	// Index tree will be stored here 
		kernels[gpuIndex][3].setArg(2, buffers[gpuIndex][5]); 

		// Kernel arguments for round3
		kernels[gpuIndex][4].setArg(0, buffers[gpuIndex][0]); 
		kernels[gpuIndex][4].setArg(1, buffers[gpuIndex][1]); 	// Index tree will be stored here 
		kernels[gpuIndex][4].setArg(2, buffers[gpuIndex][5]); 

		// Kernel arguments for round4
		kernels[gpuIndex][5].setArg(0, buffers[gpuIndex][1]); 
		kernels[gpuIndex][5].setArg(1, buffers[gpuIndex][2]); 	// Index tree will be stored here 
		kernels[gpuIndex][5].setArg(2, buffers[gpuIndex][5]);  

		// Kernel arguments for round5
		kernels[gpuIndex][6].setArg(0, buffers[gpuIndex][2]); 
		kernels[gpuIndex][6].setArg(1, buffers[gpuIndex][4]); 	// Index tree will be stored here 
		kernels[gpuIndex][6].setArg(2, buffers[gpuIndex][5]);  

		// Kernel arguments for Combine
		kernels[gpuIndex][7].setArg(0, buffers[gpuIndex][0]); 
		kernels[gpuIndex][7].setArg(1, buffers[gpuIndex][1]); 	
		kernels[gpuIndex][7].setArg(2, buffers[gpuIndex][2]); 
		kernels[gpuIndex][7].setArg(3, buffers[gpuIndex][3]); 	
		kernels[gpuIndex][7].setArg(4, buffers[gpuIndex][4]); 
		kernels[gpuIndex][7].setArg(5, buffers[gpuIndex][5]); 	
		kernels[gpuIndex][7].setArg(6, buffers[gpuIndex][6]);

		cl_int err;

		// Queue the kernels
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][0], cl::NDRange(0), cl::NDRange(12288), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][1], cl::NDRange(0), cl::NDRange(22369536), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][2], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		queues[gpuIndex].flush();

		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][3], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][4], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][5], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][6], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][7], cl::NDRange(0), cl::NDRange(4096), cl::NDRange(16), NULL, NULL);
	} 
	else 
	{	
		// Starting the 3G kernels
		
		// Kernel arguments for cleanCounter
		kernels[gpuIndex][0].setArg(0, buffers[gpuIndex][5]); 
		kernels[gpuIndex][0].setArg(1, buffers[gpuIndex][6]);

		// Kernel arguments for round0
		kernels[gpuIndex][1].setArg(0, buffers[gpuIndex][0]); 
		kernels[gpuIndex][1].setArg(1, buffers[gpuIndex][5]); 
		kernels[gpuIndex][1].setArg(2, work); 
		kernels[gpuIndex][1].setArg(3, nonce); 
		kernels[gpuIndex][1].setArg(4, (cl_uint) 0); 

		 // Kernel arguments for round1
		kernels[gpuIndex][2].setArg(0, buffers[gpuIndex][0]); 
		kernels[gpuIndex][2].setArg(1, buffers[gpuIndex][1]); 
		kernels[gpuIndex][2].setArg(2, buffers[gpuIndex][2]);  // Index tree will be stored here 
		kernels[gpuIndex][2].setArg(3, buffers[gpuIndex][5]); 
		kernels[gpuIndex][2].setArg(4, (cl_uint) 0); 

		// Kernel arguments for round2
		kernels[gpuIndex][3].setArg(0, buffers[gpuIndex][1]); 
		kernels[gpuIndex][3].setArg(1, buffers[gpuIndex][0]);	// Index tree will be stored here 
		kernels[gpuIndex][3].setArg(2, buffers[gpuIndex][5]); 

		// Kernel arguments for move
		kernels[gpuIndex][9].setArg(0, buffers[gpuIndex][2]); 
		kernels[gpuIndex][9].setArg(1, buffers[gpuIndex][1]); 	

		// Kernel arguments for repack
		kernels[gpuIndex][8].setArg(0, buffers[gpuIndex][1]); 
		kernels[gpuIndex][8].setArg(1, buffers[gpuIndex][0]); 
		kernels[gpuIndex][8].setArg(2, buffers[gpuIndex][2]); 	// Index tree will be stored here 	

		// Kernel arguments for round3
		kernels[gpuIndex][4].setArg(0, buffers[gpuIndex][0]); 
		kernels[gpuIndex][4].setArg(1, buffers[gpuIndex][1]); 	// Index tree will be stored here 
		kernels[gpuIndex][4].setArg(2, buffers[gpuIndex][5]); 

		// Kernel arguments for round4
		kernels[gpuIndex][5].setArg(0, buffers[gpuIndex][1]); 
		kernels[gpuIndex][5].setArg(1, buffers[gpuIndex][0]); 	// Index tree will be stored here 
		kernels[gpuIndex][5].setArg(2, buffers[gpuIndex][5]);  

		// Kernel arguments for round5
		kernels[gpuIndex][6].setArg(0, buffers[gpuIndex][0]); 
		kernels[gpuIndex][6].setArg(1, buffers[gpuIndex][4]); 	// Index tree will be stored here 
		kernels[gpuIndex][6].setArg(2, buffers[gpuIndex][5]);  

		// Kernel arguments for Combine
		kernels[gpuIndex][7].setArg(0, buffers[gpuIndex][1]); 
		kernels[gpuIndex][7].setArg(1, buffers[gpuIndex][2]); 	
		kernels[gpuIndex][7].setArg(2, buffers[gpuIndex][4]);  
		kernels[gpuIndex][7].setArg(3, buffers[gpuIndex][5]); 	
		kernels[gpuIndex][7].setArg(4, buffers[gpuIndex][6]);

		cl_int err;

		// Queue the kernels
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][0], cl::NDRange(0), cl::NDRange(12288), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][1], cl::NDRange(0), cl::NDRange(22369536), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][2], cl::NDRange(0), cl::NDRange(8388608), cl::NDRange(256), NULL, NULL);
		queues[gpuIndex].flush();
		
		kernels[gpuIndex][1].setArg(4, (cl_uint) 1); 
		kernels[gpuIndex][2].setArg(4, (cl_uint) 1); 
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][1], cl::NDRange(0), cl::NDRange(22369536), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][2], cl::NDRange(0), cl::NDRange(8388608), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][3], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][9], cl::NDRange(0), cl::NDRange(34799616), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][8], cl::NDRange(0), cl::NDRange(69599232), cl::NDRange(256), NULL, NULL);
		queues[gpuIndex].flush();
		
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][4], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][5], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][6], cl::NDRange(0), cl::NDRange(16777216), cl::NDRange(256), NULL, NULL);
		err = queues[gpuIndex].enqueueNDRangeKernel(kernels[gpuIndex][7], cl::NDRange(0), cl::NDRange(4096), cl::NDRange(16), NULL, NULL); 
	}	
}

void clHost::queueWork(uint32_t gpuIndex, clCallbackData* workData) 
{
	queueKernels(gpuIndex, workData);

	results[gpuIndex] = (unsigned *)queues[gpuIndex].enqueueMapBuffer(buffers[gpuIndex][6], CL_FALSE, CL_MAP_READ, 0, sizeof(cl_uint4) * 81, NULL, &events[gpuIndex], NULL);
	events[gpuIndex].setCallback(CL_COMPLETE, &CCallbackFunc, (void*) workData);
	queues[gpuIndex].flush();

	workCounter++;
}

// this function will sumit the solutions done on GPU, then fetch new work and restart mining
void clHost::callbackFunc(cl_int err , void* data)
{
	clCallbackData* workInfo = (clCallbackData*) data;
	uint32_t gpuIndex = workInfo->gpuIndex;

	// Read the number of solutions of the last iteration
	uint32_t solutions = results[gpuIndex][0];
	for (uint32_t  i = 0; i < solutions; i++) 
	{
		vector<uint32_t> indexes;
		indexes.assign(32,0);
		memcpy(indexes.data(), &results[gpuIndex][4 + 32*i], sizeof(uint32_t) * 32);

		workInfo->stratum->handleSolution(workInfo->workDescription, indexes);
	}

	solutionCnt[gpuIndex] += solutions;

	// give the GPU a breather
	this_thread::sleep_for(std::chrono::milliseconds(1000 - intensities[gpuIndex]));

	// Get new work and resume working
	if (minerStratum->hasWork()) 
	{
		currentWork[gpuIndex].stratum = minerStratum;

		queues[gpuIndex].enqueueUnmapMemObject(buffers[gpuIndex][6], results[gpuIndex], NULL, NULL);
		queueWork(gpuIndex, &currentWork[gpuIndex]);
	}
	else 
	{
		paused[gpuIndex] = true;
		
		cout << "Device will be paused, waiting for new work..." << endl;
	}
}

void clHost::startMining() 
{
	cout << endl;
	cout << "Waiting for work from stratum:" << endl;
	cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;

	minerStratum->startWorking();
	while (minerStratum->hasConnection() && !minerStratum->hasWork()) 
	{
		this_thread::sleep_for(std::chrono::milliseconds(200));
	}

	// Start mining initially
	for (size_t i = 0; i < devices.size(); i++) 
	{
		currentWork[i].gpuIndex = i;
		currentWork[i].clHost = (void*) this;

		if (minerStratum->hasWork())
		{
			paused[i] = false;
			currentWork[i].stratum = minerStratum;

			queueWork(i, &currentWork[i]);
		}
	}

	// While the mining is running print some statistics and try to wake up paused GPUs
	restart = true;
	while (restart) 
	{
			this_thread::sleep_for(std::chrono::milliseconds(15000));

		// Check if there is an active stratum connection
		if (!minerStratum->hasConnection())
		{
			restart = false;

			cout << "No connection to server. Exiting..." << endl;
		}
		else
		{
			if (!minerStratum->isConnecting())
			{
				// Print performance stats (roughly)
				cout << "Hashrate: ";
				uint32_t totalSols = 0;
				for (size_t i = 0; i < devices.size(); i++) 
				{
					uint32_t sol = solutionCnt[i];
					solutionCnt[i] = 0;
					totalSols += sol;
					cout << fixed << setprecision(2) << (double) sol / 15.0 << " sol/s ";
				}

				if (devices.size() > 1)
				{
					cout << "| Total: " << setprecision(2) << (double) totalSols / 15.0 << " sol/s ";
				}
				cout << endl;
			}
			
			// Check if there are paused devices and restart them
			for (size_t i = 0; i < devices.size(); i++) 
			{
				if (paused[i] && minerStratum->hasWork()) 
				{
					paused[i] = false;
					currentWork[i].stratum = minerStratum;

					queueWork(i, &currentWork[i]);
				}
			}
		}
	}
}

} 	// end namespace

