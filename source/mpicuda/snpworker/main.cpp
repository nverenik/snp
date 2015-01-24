#include <stdio.h>
#include <memory>

#include <mpi.h>
#include <cuda_runtime.h>

#include "snpCommand.h"
using snp::snpCommand;

#include <tclap/CmdLine.h>

struct Config;
struct DeviceInfo;

typedef std::vector<DeviceInfo> SystemInfo;

static void OnExit();
static bool ProcessCommandLine(int argc, char* argv[], Config &roConfig);
static bool GetSystemInfo(int iRank, SystemInfo &roSystemInfo);
static void PrintSystemInfo(const SystemInfo &roSystemInfo);
static void RunLoop(int iRank, const SystemInfo &roSystemInfo);

// snp commands implementation
static bool SendSystemInfo(const SystemInfo &roSystemInfo);
static bool Startup();
static bool Shutdown();
static bool Exec();
static bool Read();

struct Config
{
	bool	m_bTestEnabled;
	bool	m_bLogSystemInfo;
};

struct DeviceInfo
{
	int				m_iNodeRank;
	cudaDeviceProp	m_oProperties;
};

static const int	s_iMpiHostRank = 0;
static int			s_iMpiRank = -1;
static char			s_pszProcessorName[MPI_MAX_PROCESSOR_NAME];

#define MPI_LOG(__format__, ...) printf("[%d:%s] "__format__"\n", s_iMpiRank, s_pszProcessorName, ##__VA_ARGS__)

int main(int argc, char* argv[])
{
	::atexit(OnExit);
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &s_iMpiRank);

	int iLength = 0;
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

	RunLoop(s_iMpiRank, oSystemInfo);

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

static bool GetSystemInfo(int iRank, SystemInfo &roSystemInfo)
{
	// at first collect info about all available GPUs

	// host worker collect total number of available devices
	// prepare buffer for receiving
	int iNodeCount = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &iNodeCount);

	std::vector<int> aDeviceCount;
	aDeviceCount.resize(iNodeCount);

	// get number of available devices for the current node
	int iDeviceCount = 0;
	cudaError_t eErrorCode = cudaGetDeviceCount(&iDeviceCount);
	if (eErrorCode != cudaSuccess)
		MPI_LOG("%s", cudaGetErrorString(eErrorCode));
	else
		MPI_LOG("number of devices: %d", iDeviceCount);

	MPI_Allgather(&iDeviceCount, 1, MPI_INT, &aDeviceCount.front(), 1, MPI_INT, MPI_COMM_WORLD);

	// there're no devices on this or some else node
	for (int iNodeIndex = 0; iNodeIndex < iNodeCount; iNodeIndex++)
	{
		// TODO: node without devices sends message to host to be deleted from
		// communication group
		if (!aDeviceCount[iNodeIndex])
			return false;
	}

	if (iRank == s_iMpiHostRank)
	{
		// find the total number of devices to prepare receiver buffer
		int iDeviceCountTotal = 0;
		for (int iNodeIndex = 0; iNodeIndex < aDeviceCount.size(); iNodeIndex++)
			iDeviceCountTotal += aDeviceCount[iNodeIndex];

		MPI_LOG("Total number of devices: %d", iDeviceCountTotal);
		roSystemInfo.resize(iDeviceCountTotal);
	}

	// prepare data to send (it includes the host as well)
	SystemInfo oNodeInfo;
	// for each found GPU device create info element
	for (int iIndex = 0; iIndex < iDeviceCount; iIndex++)
	{
		oNodeInfo.push_back(DeviceInfo());
		DeviceInfo *pDeviceInfo = &oNodeInfo[iIndex];

		pDeviceInfo->m_iNodeRank = iRank;
		cudaError_t eErrorCode = cudaGetDeviceProperties(&pDeviceInfo->m_oProperties, iIndex);
		if (eErrorCode != cudaSuccess)
			MPI_LOG("%s", cudaGetErrorString(eErrorCode));
	}

	// threat device info struct as raw array of bytes
	MPI_Datatype iDeviceInfoType;
	MPI_Type_contiguous(sizeof(DeviceInfo), MPI_BYTE, &iDeviceInfoType);
	MPI_Type_commit(&iDeviceInfoType);

	// collect all info on the host
	MPI_Gather(&oNodeInfo.front(), int(oNodeInfo.size()), iDeviceInfoType, &roSystemInfo.front(), int(oNodeInfo.size()), iDeviceInfoType, s_iMpiHostRank, MPI_COMM_WORLD);
	MPI_Type_free(&iDeviceInfoType);

	return true;
}

static void PrintSystemInfo(const SystemInfo &roSystemInfo)
{
	int iNodeRank = -1;
	for (int iDeviceIndex = 0; iDeviceIndex < roSystemInfo.size(); iDeviceIndex++)
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

static void RunLoop(int iRank, const SystemInfo &roSystemInfo)
{
	MPI_LOG("Process started...");
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
			case snp::STARTUP:		Startup();		break;
			case snp::SHUTDOWN:		Shutdown();		break;
			case snp::EXEC:			Exec();			break;
			case snp::READ:			Read();			break;

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
	return false;
}

static bool Startup()
{
	return false;
}

static bool Shutdown()
{
	return false;
}

static bool Exec()
{
	return false;
}

static bool Read()
{
	return false;
}