#include <snp/snpDevice.h>
#include "snpBackendConfig.h"

#if (SNP_TARGET_BACKEND == SNP_BACKEND_CUDA)

#include "cuda_runtime.h"
#include "kernel.h"

#include <string>

NS_SNP_BEGIN

#define snpBreakIf(__statement__, __message__) { \
	if (__statement__) { \
		fprintf(stderr, "snp-cuda ERROR: %s\n", (__message__)); \
		break; \
	} \
}

#define snpBreakIfNot(__statement__, __message__) { \
	if (!(__statement__)) { \
		fprintf(stderr, "snp-cuda ERROR: %s\n", (__message__)); \
		break; \
	} \
}

#define snpCheckCuda(__statement__) { \
	cudaError_t ret_cuda = (__statement__); \
	snpBreakIf(ret_cuda != cudaSuccess, cudaGetErrorString(ret_cuda)); \
}


#define snpLogError(__message__) { \
	fprintf(stderr, "snp-cuda ERROR: %s\n", (__message__)); \
}

#define snpCudaSafeCall(__statement__) { \
	cudaError_t ret_cuda = (__statement__); \
	if (ret_cuda != cudaSuccess) { \
		snpLogError(cudaGetErrorString(ret_cuda)); \
	} \
}

static uint32	*d_memory;
static uint32	*d_instruction;
static int32	*d_output;

static int32	*h_output;
static uint32	*h_cell;

static uint32	s_cellIndex;

void ChildNodeProcess(uint32 rank);

//bool CheckResourceSufficiency(int iCellSize, int iNumCellsPerPU, int iNumPUs)
//{
//	int iDeviceCount;
//	cudaGetDeviceCount(&iDeviceCount);
//
//	struct cudaDeviceProp oDeviceProp;
//	cudaGetDeviceProperties(&oDeviceProp, 0); // We assume that system has similar devices
//
//	// Check threads
//	int iMaxThreads = iDeviceCount * oDeviceProp.maxGridSize[0] * oDeviceProp.maxThreadsDim[0];
//	if(iMaxThreads < iNumPUs)
//	{
//		printf("Threads check failed");
//		return false;
//	}
//
//	// Check memory
//	unsigned long iTotalGlobalMemory = iDeviceCount * oDeviceProp.totalGlobalMem;
//	unsigned long iRequiredMemory = ( iCellSize * iNumCellsPerPU * iNumPUs * sizeof(int) ) + ( 4 * iCellSize * sizeof(int) ) + ( iNumPUs * sizeof(int) );
//
//	if(iTotalGlobalMemory < iRequiredMemory)
//	{
//		printf("Memory check failed");
//		return false;
//	}
//
//	return true;
//}
//
//bool snpDeviceImpl::systemInfo()
//{
//	int iDeviceCount;
//	cudaGetDeviceCount(&iDeviceCount);
//
//	struct cudaDeviceProp oDeviceProp;
//	std::string sDeviceName;
//	for (int iDevice = 0; iDevice < iDeviceCount; iDevice++)
//	{
//		cudaGetDeviceProperties(&oDeviceProp, iDevice);
//
//		if(!sDeviceName.empty() && sDeviceName != (std::string)oDeviceProp.name)
//			return false;
//
//		sDeviceName = oDeviceProp.name;
//	}
//
//	// System properties:
//	printf("Device count: %d\n", iDeviceCount);
//	printf("\n");
//
//	// General
//	printf("Device name: %s\n", oDeviceProp.name);
//	printf("Clock rate: %d\n", oDeviceProp.clockRate);
//	printf("\n");
//
//	// Computing
//	printf("Multiprocessor count: %d\n", oDeviceProp.multiProcessorCount);
//	printf("Max Threads per Multiprocessor: %d\n", oDeviceProp.maxThreadsPerMultiProcessor);
//	printf("Max grid size (x;y;z): (%d;%d;%d)\n",
//		oDeviceProp.maxGridSize[0],
//		oDeviceProp.maxGridSize[1],
//		oDeviceProp.maxGridSize[2]);
//	printf("Max block size (x;y;z): (%d;%d;%d)\n",
//		oDeviceProp.maxThreadsDim[0],
//		oDeviceProp.maxThreadsDim[1],
//		oDeviceProp.maxThreadsDim[2]);
//	printf("Warp size: %d\n", oDeviceProp.warpSize);
//	printf("\n");
//
//	// Memory
//	printf("Total global memory: %d\n", oDeviceProp.totalGlobalMem);
//	printf("Total constant memory: %d\n", oDeviceProp.totalConstMem);
//	printf("Shared memory per block: %d\n", oDeviceProp.sharedMemPerBlock);
//	printf("Registers per block: %d\n", oDeviceProp.regsPerBlock);
//	printf("\n");
//
//	// Recommendations
//	printf("Min threads: %d\n", iDeviceCount * oDeviceProp.multiProcessorCount * oDeviceProp.maxThreadsPerMultiProcessor);	// Num of threads that may run at once
//	printf("Max threads: %d\n", iDeviceCount * oDeviceProp.maxGridSize[0] * oDeviceProp.maxThreadsDim[0]);					// Logical limit
//	printf("Memory for Min: %d\n", oDeviceProp.totalGlobalMem / (oDeviceProp.multiProcessorCount * oDeviceProp.maxThreadsPerMultiProcessor));
//	printf("Memory for Max: %d\n", oDeviceProp.totalGlobalMem / (oDeviceProp.maxGridSize[0] * oDeviceProp.maxThreadsDim[0]));
//
//	return true;
//}

bool snpDeviceImpl::init(uint16 cellSize, uint32 cellsPerPU, uint32 numberOfPU)
{
	// 1. MPI - get number of available nodes
	// ...

	// 2. MPI - get rank for current process

	// 2. MPI - request

	/////////////////////////


	//if (!CheckResourceSufficiency(cellSize, cellsPerPU, numberOfPU))
	//	return false;

 //   d_memory = nullptr;
 //   d_instruction = nullptr;
 //   d_output = nullptr;
 //   h_output = nullptr;
 //   h_cell = nullptr;
	//s_cellIndex = kCellNotFound;

	//m_cellSize = cellSize;
	//m_cellsPerPU = cellsPerPU;
	//m_numberOfPU = numberOfPU;

	//do
	//{
	//	snpBreakIfNot(cellSize > 0 && cellsPerPU > 0 && numberOfPU > 0, "snpDeviceImpl::init() - invalid device configuration.");
	//	const uint32 memorySize = cellSize * cellsPerPU * numberOfPU;

	//	// todo: check if memory is available
	//	// ...

	//	// initialize video adapter
	//	snpCheckCuda(cudaSetDevice(0));

	//	// allocate processor memory in GPU
	//	snpCheckCuda(cudaMalloc((void**)&d_memory, memorySize * sizeof(uint32)));
	//	snpCheckCuda(cudaMalloc((void**)&d_instruction, 4 * cellSize * sizeof(uint32)));
	//	snpCheckCuda(cudaMalloc((void**)&d_output, numberOfPU * sizeof(int32)));

	//	// allocate buffer memory for output array
	//	h_output = new int32[numberOfPU];
	//	// allocate buffer memory for single cell
	//	h_cell = new uint32[cellSize];

	//	return true;
	//}
	//while(false);
	//return false;
}

void snpDeviceImpl::deinit()
{
	//snpCudaSafeCall(cudaFree(d_memory));
	//snpCudaSafeCall(cudaFree(d_instruction));
	//snpCudaSafeCall(cudaFree(d_output));
	//snpCudaSafeCall(cudaDeviceReset());

	//delete h_output;
	//delete h_cell;
}

bool snpDeviceImpl::exec(bool singleCell, snpOperation operation, const uint32 * const instruction)
{
	//s_cellIndex = kernel_exec(singleCell, operation, instruction, m_cellSize, m_cellsPerPU, m_numberOfPU, d_memory, d_instruction, d_output, h_output, h_cell);
	//return (s_cellIndex != kCellNotFound);
}

bool snpDeviceImpl::read(uint32 *bitfield)
{
	//if (s_cellIndex != kCellNotFound)
	//{
	//	snpCudaSafeCall(cudaMemcpy(bitfield, d_memory + s_cellIndex * m_cellSize, m_cellSize * sizeof(uint32), cudaMemcpyDeviceToHost));
	//	return true;
	//}
	//return false;
}

void snpDeviceImpl::dump()
{
	//const uint32 memorySize = m_cellSize * m_cellsPerPU * m_numberOfPU;
	//uint32 *buffer = new uint32[memorySize * sizeof(uint32)];
	//snpCudaSafeCall(cudaMemcpy(buffer, d_memory, memorySize * sizeof(uint32), cudaMemcpyDeviceToHost));

	//for (uint32 cellIndex = 0; cellIndex < m_cellsPerPU; cellIndex++)
	//{
	//	for (uint32 puIndex = 0; puIndex < m_numberOfPU; puIndex++)
	//	{
	//		for (uint32 index = 0; index < m_cellSize; index++)
	//		{
	//			printf("%d ", buffer[(cellIndex * m_numberOfPU + puIndex) * m_cellSize + index]);
	//		}
	//		printf("  ");
	//	}
	//	printf("\n");
	//}

	//delete buffer;
}

// the entry point for process in the node
void ChildNodeProcess(uint32 rank)
{
}

NS_SNP_END

#endif //(SNP_TARGET_BACKEND == SNP_BACKEND_CUDA)
