#include <snp-cuda\snpDevice.h>
#include "cuda_runtime.h"

#include "kernel.h"

NS_SNP_BEGIN

#define snpBreakIf(__statement__, __message__) { \
	if (__statement__) { \
		fprintf(stderr, "snp-cuda ERROR: %s\n", (__message__)); \
		break; \
	} \
}

#define snpCheckCuda(__statement__) { \
	cudaError_t ret_cuda = (__statement__); \
	snpBreakIf(ret_cuda != cudaSuccess, cudaGetErrorString(ret_cuda)); \
}

bool snpDeviceImpl::m_exists = false;

snpDeviceImpl * snpDeviceImpl::create(uint16 cellSize, uint32 cellsPerPU, uint32 numberOfPU)
{
	snpDeviceImpl *device = (m_exists != true) ? new snpDeviceImpl() : nullptr;
	if (device != nullptr && device->init(cellSize, cellsPerPU, numberOfPU) != true)
	{
		delete device;
		device = nullptr;
	}
	return device;
}

bool snpDeviceImpl::init(uint16 cellSize, uint32 cellsPerPU, uint32 numberOfPU)
{
	m_cellSize = cellSize;
	m_cellsPerPU = cellsPerPU;
	m_numberOfPU = numberOfPU;

	do
	{
		return true;
	}
	while(false);
	return false;

	//do
	//{
	//	// are input params valid
	//	SNP_BREAK_IF(cellsPerPU == 0 || numberOfPU == 0, "snpDevice::init() - device parameters can't be 0");

	//	// calculate memory size in bytes
	//	const uint32 totalNumberOfCells = cellsPerPU * numberOfPU;
	//	const uint32 memorySize = totalNumberOfCells * sizeof(snpBitfield);
	//	SNP_BREAK_IF(memorySize > this->getMaxMemorySize(), "snpDevice::init() - not enough memory is available in GPU");

	//	// initialize video adapter
	//	SNP_CHECK_CUDA(cudaSetDevice(0));

	//	// allocate processor memory in GPU
	//	SNP_CHECK_CUDA(cudaMalloc((void**)&d_memory, memorySize));
	//	SNP_CHECK_CUDA(cudaMalloc((void**)&d_output, numberOfPU * sizeof(int32)));
	//	
	//	return true;
	//}
	//while(false);
	//return false;
}

snpDeviceImpl::snpDeviceImpl()
	: m_cellSize(0)
	, m_cellsPerPU(0)
	, m_numberOfPU(0)
{
	m_exists = true;
}

snpDeviceImpl::~snpDeviceImpl()
{
	m_exists = false;
}

NS_SNP_END
