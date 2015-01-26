#include <stdio.h>
#include <memory>

#include <tclap/CmdLine.h>

#include <mpi.h>
#include <cuda_runtime.h>

#include <snp/snpOperation.h>
using snp::snpOperation;

#include "../snpCommand.h"
using snp::snpCommand;

#include "kernel.h"

struct Config;
struct DeviceInfo;
struct DeviceConfiguration;

typedef std::vector<DeviceInfo> SystemInfo;
typedef std::vector<DeviceConfiguration> SystemConfiguration;

static void OnExit();
static bool ProcessCommandLine(int32 argc, char* argv[], Config &roConfig);
static bool GetSystemInfo(int32 iRank, SystemInfo &roSystemInfo);
static void PrintSystemInfo(const SystemInfo &roSystemInfo);
static void RunLoop(int32 iRank, const SystemInfo &roSystemInfo);

// snp commands implementation
static bool SendSystemInfo(const SystemInfo &roSystemInfo);
static bool Startup(int32 iRank, const SystemInfo &roSystemInfo, uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU);
static bool Shutdown();
static bool Exec(int32 iRank, bool bSingleCell, snpOperation eOperation, const uint32 * const pInstruction);
static bool Read(int32 iRank, uint32 *pBitfield);

struct Config
{
	bool	m_bTestEnabled;
	bool	m_bLogSystemInfo;
};

struct DeviceInfo
{
	int32			m_iNodeRank;
	cudaDeviceProp	m_oProperties;
};

struct DeviceConfiguration
{
	uint32	m_uiGridDim;	// number of blocks (1 .. 65536) within grid
	uint32	m_uiBlockDim;	// number of threads (1 .. 1024) within block
	uint32	m_uiThreadDim;	// number of cells within thread
	uint32	m_uiCellDim;	// number of uint32 within cell
};

static const int32	s_iMpiHostRank	= 0;
static int32		s_iMpiRank		= -1;
static char			s_pszProcessorName[MPI_MAX_PROCESSOR_NAME];

static SystemInfo			s_oNodeInfo;			// info about all available devices in the current node
static SystemConfiguration	s_oNodeConfiguration;	// configuration for each device in the current node

// pointers to the memory allocated on device
static std::vector<uint32 *>	d_aMemory;
static std::vector<uint32 *>	d_aInstruction;
static std::vector<int32 *>		d_aOutput;

// pre-allocated buffers used when working with kernel
static std::vector<int32 *>		h_aOutput;
static std::vector<uint32 *>	h_aCell;

// result of ther last performed Exec() command is the index of device and index of the cell inside its memory
// pointing to the first matched cell during instruction
static int32	s_iNodeIndex	= kCellNotFound;
static int32	s_iDeviceIndex	= kCellNotFound;
static int32	s_iCellIndex	= kCellNotFound;

#define MPI_LOG(__format__, ...) printf("[%d:%s] "__format__"\n", s_iMpiRank, s_pszProcessorName, ##__VA_ARGS__)

int main(int argc, char* argv[])
{
	::atexit(OnExit);
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &s_iMpiRank);

	int32 iLength = 0;
	MPI_Get_processor_name(s_pszProcessorName, &iLength);

	Config oConfig;
	if (s_iMpiRank == s_iMpiHostRank && !ProcessCommandLine(argc, argv, oConfig))
		return 0;

	SystemInfo oSystemInfo;
	if (!GetSystemInfo(s_iMpiRank, oSystemInfo))
	{
		if (s_iMpiRank == s_iMpiHostRank)
			MPI_LOG("Cannot detect available GPU device on some of worker nodes.");
		return 0;
	}

	if (s_iMpiRank == s_iMpiHostRank && oConfig.m_bLogSystemInfo)
	{
		PrintSystemInfo(oSystemInfo);
		return 0;
	}

	Startup(s_iMpiRank, oSystemInfo, 9, 32, 10000);
	Shutdown();
	//RunLoop(s_iMpiRank, oSystemInfo);

	// MPI_Finalize() is called using atexit callback
	return 0;
}

static void OnExit()
{
	MPI_Finalize();
}

static bool ProcessCommandLine(int argc, char* argv[], Config &roConfig)
{
	try
	{
		TCLAP::CmdLine oCommandLine("The worker executable is a part of software imitation model of the associative SNP "
									"(Semantic Network Processor, see http://github.com/nverenik/snp). It's responsible "
									"for connection with the main application and execution received commands on the "
									"computation cluster using MPI and NVidia CUDA frameworks.", ' ', "0.1.0");
		TCLAP::SwitchArg oTestSwitch("", "test", "Runs host worker in the test mode. It will generate a sequence of dummy "
												 "commands and send them to the nodes.", oCommandLine, false);
		TCLAP::SwitchArg oInfoSwitch("", "info", "Displays detailed cluster information and exits.", oCommandLine, false);
		
		// Parse the argv array.
		oCommandLine.parse(argc, argv);
		roConfig.m_bTestEnabled = oTestSwitch.getValue();
		roConfig.m_bLogSystemInfo = oInfoSwitch.getValue();
	}
	catch (...)
	{
		return false;
	}
	return true;
}

static bool GetSystemInfo(int32 iRank, SystemInfo &roSystemInfo)
{
	// at first collect info about all available GPUs

	// host worker collect total number of available devices
	// prepare buffer for receiving
	int32 iNodeCount = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &iNodeCount);

	std::vector<int32> aDeviceCount;
	aDeviceCount.resize(iNodeCount);

	// get number of available devices for the current node
	int32 iDeviceCount = 0;
	cudaError_t eErrorCode = cudaGetDeviceCount(&iDeviceCount);
	if (eErrorCode != cudaSuccess)
		MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
	else
		MPI_LOG("Number of devices: %d", iDeviceCount);

	MPI_Allgather(&iDeviceCount, 1, MPI_INT, &aDeviceCount.front(), 1, MPI_INT, MPI_COMM_WORLD);

	// there're no devices on this or some else node
	for (int32 iNodeIndex = 0; iNodeIndex < iNodeCount; iNodeIndex++)
	{
		// TODO: node without devices sends message to host to be deleted from
		// communication group
		if (!aDeviceCount[iNodeIndex])
			return false;
	}

	if (iRank == s_iMpiHostRank)
	{
		// find the total number of devices to prepare receiver buffer
		uint32 iDeviceCountTotal = 0;
		for (int32 iNodeIndex = 0; iNodeIndex < aDeviceCount.size(); iNodeIndex++)
			iDeviceCountTotal += aDeviceCount[iNodeIndex];

		MPI_LOG("Total number of devices: %u", iDeviceCountTotal);
		roSystemInfo.resize(iDeviceCountTotal);
	}

	// prepare data to send (it includes the host as well)
	s_oNodeInfo.clear();

	// for each found GPU device create info element
	for (int32 iIndex = 0; iIndex < iDeviceCount; iIndex++)
	{
		s_oNodeInfo.push_back(DeviceInfo());
		DeviceInfo *pDeviceInfo = &s_oNodeInfo[iIndex];

		pDeviceInfo->m_iNodeRank = iRank;
		cudaError_t eErrorCode = cudaGetDeviceProperties(&pDeviceInfo->m_oProperties, iIndex);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
	}

	// threat device info struct as raw array of bytes
	MPI_Datatype iDeviceInfoType;
	MPI_Type_contiguous(sizeof(DeviceInfo), MPI_BYTE, &iDeviceInfoType);
	MPI_Type_commit(&iDeviceInfoType);

	// TODO: seems MPI_Gather do not allow to send data with different size, replace it with MPI_Send for each node
	// collect all info on the host
	MPI_Gather(&s_oNodeInfo.front(), int32(s_oNodeInfo.size()), iDeviceInfoType, &roSystemInfo.front(), int32(s_oNodeInfo.size()), iDeviceInfoType, s_iMpiHostRank, MPI_COMM_WORLD);
	MPI_Type_free(&iDeviceInfoType);

	return true;
}

static void PrintSystemInfo(const SystemInfo &roSystemInfo)
{
	int32 iNodeRank = -1;
	for (int32 iDeviceIndex = 0; iDeviceIndex < roSystemInfo.size(); iDeviceIndex++)
	{
		const DeviceInfo &roDeviceInfo = roSystemInfo[iDeviceIndex];
		if (roDeviceInfo.m_iNodeRank != iNodeRank)
		{
			printf("\n");
			printf("=============== Node #%d ===============\n", roDeviceInfo.m_iNodeRank);
			iNodeRank = roDeviceInfo.m_iNodeRank;
		}

		const cudaDeviceProp &roDeviceProps = roDeviceInfo.m_oProperties;

		printf("\n");
		printf("[Device #%d]", iDeviceIndex);
		printf("\n");

		// General
		printf("   Device name: %s\n", roDeviceProps.name);
		printf("   Clock rate: %d\n", roDeviceProps.clockRate);
		printf("\n");

		// Computing
		printf("   Multiprocessor count: %d\n", roDeviceProps.multiProcessorCount);
		printf("   Max Threads per Multiprocessor: %d\n", roDeviceProps.maxThreadsPerMultiProcessor);
		printf("   Max grid size (x;y;z): (%d;%d;%d)\n",
			roDeviceProps.maxGridSize[0],
			roDeviceProps.maxGridSize[1],
			roDeviceProps.maxGridSize[2]);
		printf("   Max block size (x;y;z): (%d;%d;%d)\n",
			roDeviceProps.maxThreadsDim[0],
			roDeviceProps.maxThreadsDim[1],
			roDeviceProps.maxThreadsDim[2]);
		printf("   Warp size: %d\n", roDeviceProps.warpSize);
		printf("\n");

		// Memory
		printf("   Total global memory: %d\n", roDeviceProps.totalGlobalMem);
		printf("   Total constant memory: %d\n", roDeviceProps.totalConstMem);
		printf("   Shared memory per block: %d\n", roDeviceProps.sharedMemPerBlock);
		printf("   Registers per block: %d\n", roDeviceProps.regsPerBlock);
		printf("\n");
	}

	// TODO: review these recomendations
	//// Recommendations
	//printf("   Min threads: %d\n", iDeviceCount * roDeviceProps.multiProcessorCount * roDeviceProps.maxThreadsPerMultiProcessor);	// Num of threads that may run at once
	//printf("   Max threads: %d\n", iDeviceCount * roDeviceProps.maxGridSize[0] * roDeviceProps.maxThreadsDim[0]);					// Logical limit
	//printf("   Memory for Min: %d\n", roDeviceProps.totalGlobalMem / (roDeviceProps.multiProcessorCount * roDeviceProps.maxThreadsPerMultiProcessor));
	//printf("   Memory for Max: %d\n", roDeviceProps.totalGlobalMem / (roDeviceProps.maxGridSize[0] * roDeviceProps.maxThreadsDim[0]));
}

static void RunLoop(int32 iRank, const SystemInfo &roSystemInfo)
{
	MPI_LOG("Process running...");
	while(true)
	{
		snpCommand eCommand = snp::Undefined;
		if (iRank == s_iMpiHostRank)
		{
			// TODO: read command from the socket
			while(true)
			{
				// eCommand = ??
				if (eCommand == snp::SYSTEM_INFO)
				{
					SendSystemInfo(roSystemInfo);
					continue;
				}

				// TODO: break when eCommand is ready
			}
		}

		// broadcast command to all nodes
		MPI_Bcast(&eCommand, 1, MPI_INT, s_iMpiHostRank, MPI_COMM_WORLD);

		// perform command 
		switch(eCommand)
		{
			case snp::STARTUP:		Startup(iRank, roSystemInfo, 9, 32, 32); break;	// TODO: check double call before broadcasting startup command
			case snp::SHUTDOWN:		Shutdown();		break;	// TODO: check double call
			case snp::EXEC:			/*Exec();*/			break;
			case snp::READ:			/*Read();*/			break;	// TODO: preallocate buffer for reading

			case snp::SYSTEM_INFO:
			default:
			{
				break;
			}
		}
	}
}

static bool SendSystemInfo(const SystemInfo &roSystemInfo)
{
	// TODO: send system info to the main process
	// for now this command is not used
	return false;
}

static bool Startup(int32 iRank, const SystemInfo &roSystemInfo, uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU)
{
	assert(!d_aMemory.size());
	assert(!d_aInstruction.size());
	assert(!d_aOutput.size());

	// Find the configuration for GPUs
	SystemConfiguration oSystemConfiguration;

	// (!)Assume that each device (GPU) is used maximum only by only one node
	if (iRank == s_iMpiHostRank)
	{
		// 1. number of blocks is multiple of multiprocessors amount
		// 2. as number of threads per block use the maximum
		uint32 uiNumberOfThreadsPerIteration = 0;
		for (int32 iDeviceIndex = 0; iDeviceIndex < roSystemInfo.size(); iDeviceIndex++)
		{
			// for each iteration add number of threads equals to what we obtain if add
			// 1 block for each multi processor and use maximum threads in this block
			uiNumberOfThreadsPerIteration += 
				roSystemInfo[iDeviceIndex].m_oProperties.multiProcessorCount *
				roSystemInfo[iDeviceIndex].m_oProperties.maxThreadsDim[0];
		}

		// find the minimum configuration which covers requested memory volume
		uint32 uiMultiplier = uint32(ceilf((float)uiNumberOfPU / uiNumberOfThreadsPerIteration));
		for (int32 iDeviceIndex = 0; iDeviceIndex < roSystemInfo.size(); iDeviceIndex++)
		{
			oSystemConfiguration.push_back(DeviceConfiguration());
			DeviceConfiguration *pDeviceConfiguration = &oSystemConfiguration[iDeviceIndex];

			pDeviceConfiguration->m_uiCellDim = uiCellSize;
			pDeviceConfiguration->m_uiThreadDim = uiCellsPerPU;
			pDeviceConfiguration->m_uiBlockDim = roSystemInfo[iDeviceIndex].m_oProperties.maxThreadsDim[0];
			pDeviceConfiguration->m_uiGridDim = roSystemInfo[iDeviceIndex].m_oProperties.multiProcessorCount * uiMultiplier;
		}
	}

	// send configurations to each process
	const uint32 uiNumberOfLocalDevices = uint32(s_oNodeInfo.size());

	s_oNodeConfiguration.clear();
	s_oNodeConfiguration.resize(uiNumberOfLocalDevices);

	// threat device info struct as raw array of bytes
	MPI_Datatype iDeviceConfigurationType;
	MPI_Type_contiguous(sizeof(DeviceConfiguration), MPI_BYTE, &iDeviceConfigurationType);
	MPI_Type_commit(&iDeviceConfigurationType);

	// TODO: seems MPI_Scatter do not allow to send data with different size, replace it with MPI_Send for each node
	// collect all info on the host
	MPI_Scatter(&oSystemConfiguration.front(), int32(oSystemConfiguration.size()), iDeviceConfigurationType,
		&s_oNodeConfiguration.front(), uiNumberOfLocalDevices, iDeviceConfigurationType, s_iMpiHostRank, MPI_COMM_WORLD);
	MPI_Type_free(&iDeviceConfigurationType);

	// allocate processor memory separately for each GPU
	d_aMemory.resize(uiNumberOfLocalDevices);
	d_aInstruction.resize(uiNumberOfLocalDevices);
	d_aOutput.resize(uiNumberOfLocalDevices);

	h_aOutput.resize(uiNumberOfLocalDevices);
	h_aCell.resize(uiNumberOfLocalDevices);

	cudaError_t eErrorCode = cudaSuccess;
	for (uint32 iDeviceIndex = 0; iDeviceIndex < uiNumberOfLocalDevices; iDeviceIndex++)
	{
		eErrorCode = cudaSetDevice(iDeviceIndex);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		const DeviceConfiguration &roDeviceConfiguration = s_oNodeConfiguration[iDeviceIndex];
		const uint32 uiMemorySize =
			roDeviceConfiguration.m_uiCellDim *
			roDeviceConfiguration.m_uiThreadDim *
			roDeviceConfiguration.m_uiBlockDim *
			roDeviceConfiguration.m_uiGridDim;

		MPI_LOG("Configure device #%d", iDeviceIndex);
		MPI_LOG("   Cell dim = %u", roDeviceConfiguration.m_uiCellDim);
		MPI_LOG("   Thread dim = %u", roDeviceConfiguration.m_uiThreadDim);
		MPI_LOG("   Block dim = %u", roDeviceConfiguration.m_uiBlockDim);
		MPI_LOG("   Grid dim = %u", roDeviceConfiguration.m_uiGridDim);
		MPI_LOG("Memory allocated = %u", uiMemorySize * sizeof(uint32));

		eErrorCode = cudaMalloc((void**)&d_aMemory[iDeviceIndex], uiMemorySize * sizeof(uint32));
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		// TODO: should we place instruction and output arrays into shared memory or something for speadup?
		eErrorCode = cudaMalloc((void**)&d_aInstruction[iDeviceIndex], 4 * roDeviceConfiguration.m_uiCellDim * sizeof(uint32));
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		eErrorCode = cudaMalloc((void**)&d_aOutput[iDeviceIndex], roDeviceConfiguration.m_uiBlockDim * roDeviceConfiguration.m_uiGridDim * sizeof(int32));
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		// allocate buffer memory for output array
		h_aOutput[iDeviceIndex] = new int32[roDeviceConfiguration.m_uiBlockDim * roDeviceConfiguration.m_uiGridDim];
		h_aCell[iDeviceIndex] = new uint32[roDeviceConfiguration.m_uiCellDim];
	}

	if (iRank == s_iMpiHostRank)
	{
		uint32 uiTotalNumberOfPU = 0;
		for (int32 iDeviceIndex = 0; iDeviceIndex < oSystemConfiguration.size(); iDeviceIndex++)
		{
			uiTotalNumberOfPU +=
				oSystemConfiguration[iDeviceIndex].m_uiBlockDim * 
				oSystemConfiguration[iDeviceIndex].m_uiGridDim;
		}
		MPI_LOG("Total number of PU: %u", uiTotalNumberOfPU);
	}

	return true;
}

static bool Shutdown()
{
	cudaError_t eErrorCode = cudaSuccess;
	for (int32 iDeviceIndex = 0; iDeviceIndex < s_oNodeConfiguration.size(); iDeviceIndex++)
	{
		eErrorCode = cudaSetDevice(iDeviceIndex);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		eErrorCode = cudaFree(d_aMemory[iDeviceIndex]);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		eErrorCode = cudaFree(d_aInstruction[iDeviceIndex]);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		eErrorCode = cudaFree(d_aOutput[iDeviceIndex]);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		eErrorCode = cudaDeviceReset();
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		delete h_aOutput[iDeviceIndex];
		delete h_aCell[iDeviceIndex];
	}

	d_aMemory.clear();
	d_aInstruction.clear();
	d_aOutput.clear();

	h_aOutput.clear();
	h_aCell.clear();

	MPI_LOG("System shutdown.");
	return true;
}

static bool Exec(int32 iRank, bool bSingleCell, snpOperation eOperation, const uint32 * const pInstruction)
{
	// execute kernel function for each device
	s_iDeviceIndex = kCellNotFound;
	for (int32 iDeviceIndex = 0; iDeviceIndex < s_oNodeConfiguration.size(); iDeviceIndex++)
	{
		cudaError_t eErrorCode = cudaSetDevice(iDeviceIndex);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		const DeviceConfiguration &roDeviceConfiguration = s_oNodeConfiguration[iDeviceIndex];
		s_iCellIndex = kernel_exec(
			bSingleCell,
			eOperation,
			pInstruction,
			roDeviceConfiguration.m_uiCellDim,
			roDeviceConfiguration.m_uiThreadDim,
			roDeviceConfiguration.m_uiBlockDim,
			roDeviceConfiguration.m_uiGridDim,
			d_aMemory[iDeviceIndex],
			d_aInstruction[iDeviceIndex],
			d_aOutput[iDeviceIndex],
			h_aOutput[iDeviceIndex],
			h_aCell[iDeviceIndex]
		);

		if (s_iCellIndex != kCellNotFound)
		{
			s_iDeviceIndex = iDeviceIndex;
			break;
		}
	}

	// share with host just the fact that cell is found
	bool bFound = (s_iDeviceIndex != kCellNotFound && s_iCellIndex != kCellNotFound);

	// prepare buffer to receive
	int32 iGroupSize = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &iGroupSize);

	std::vector<bool> aFound;
	if (iRank == s_iMpiHostRank)
		aFound.resize(iGroupSize);

	// share result
	MPI_Gather(&bFound, 1, MPI_BYTE, &aFound.front(), 1, MPI_BYTE, s_iMpiHostRank, MPI_COMM_WORLD);

	// find the first matched node
	if (iRank == s_iMpiHostRank)
	{
		s_iNodeIndex = kCellNotFound;
		for (int32 iNodeIndex = 0; iNodeIndex < aFound.size(); iNodeIndex++)
		{
			if (aFound[iNodeIndex])
			{
				s_iNodeIndex = iNodeIndex;
				break;
			}
		}
		return (s_iNodeIndex != kCellNotFound);
	}
	return bFound;
}

static bool Read(int32 iRank, uint32 *pBitfield)
{
	// broadcast send index if target node
	int32 iNodeIndex = kCellNotFound;
	MPI_Bcast(&iNodeIndex, 1, MPI_INT, s_iMpiHostRank, MPI_COMM_WORLD);

	if (iNodeIndex == kCellNotFound)
		return false;
	
	// host node can store data itself
	if (iRank == iNodeIndex)
	{
		assert(s_iDeviceIndex != kCellNotFound);
		assert(s_iCellIndex != kCellNotFound);

		// activate selected device (did store in the last exec() call)
		cudaError_t eErrorCode = cudaSetDevice(s_iDeviceIndex);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		// get data directly from device
		const DeviceConfiguration &roDeviceConfiguration = s_oNodeConfiguration[s_iDeviceIndex];
		eErrorCode = cudaMemcpy(
			pBitfield,
			d_aMemory[s_iDeviceIndex] + s_iCellIndex * roDeviceConfiguration.m_uiCellDim,
			roDeviceConfiguration.m_uiCellDim * sizeof(uint32),
			cudaMemcpyDeviceToHost
		);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));

		// send data if needed
		if (iRank != s_iMpiHostRank)
			MPI_Send(pBitfield, roDeviceConfiguration.m_uiCellDim * sizeof(uint32), MPI_BYTE, s_iMpiHostRank, MPI_ANY_TAG, MPI_COMM_WORLD);
	}

	if (iRank == s_iMpiHostRank && iRank != iNodeIndex)
	{
		int32 iSize = s_oNodeConfiguration[0].m_uiCellDim * sizeof(uint32);
		MPI_Recv(pBitfield, iSize, MPI_BYTE, iNodeIndex, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	return true;
}