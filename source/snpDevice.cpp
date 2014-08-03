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

snpDeviceImpl::snpDeviceImpl()
	: m_cellSize(0)
	, m_cellsPerPU(0)
	, m_numberOfPU(0)
	, d_memory(nullptr)
	, d_output(nullptr)
	, h_output(nullptr)
	, h_cell(nullptr)
	, m_cellIndex(kCellNotFound)
{
	m_exists = true;
}

snpDeviceImpl::~snpDeviceImpl()
{
	snpCudaSafeCall(cudaFree(d_memory));
	snpCudaSafeCall(cudaFree(d_output));
	snpCudaSafeCall(cudaDeviceReset());

	delete h_output;
	delete h_cell;

	m_exists = false;
}

bool snpDeviceImpl::init(uint16 cellSize, uint32 cellsPerPU, uint32 numberOfPU)
{
	m_cellSize = cellSize;
	m_cellsPerPU = cellsPerPU;
	m_numberOfPU = numberOfPU;

	do
	{
		snpBreakIfNot(cellSize > 0 && cellsPerPU > 0 && numberOfPU > 0, "snpDeviceImpl::init() - invalid device configuration.");
		const uint32 memorySize = cellSize * cellsPerPU * numberOfPU;

		// todo: check if memory is available
		// ...

		// initialize video adapter
		snpCheckCuda(cudaSetDevice(0));

		// allocate processor memory in GPU
		snpCheckCuda(cudaMalloc((void**)&d_memory, memorySize * sizeof(uint32)));
		snpCheckCuda(cudaMalloc((void**)&d_output, numberOfPU * sizeof(int32)));

		// allocate buffer memory for output array
		h_output = new int32[numberOfPU];
		// allocate buffer memory for single cell
		h_cell = new uint32[cellSize];

		return true;
	}
	while(false);
	return false;
}

bool snpDeviceImpl::exec(bool singleCell, snpOperation operation, const uint32 * const instruction)
{
	m_cellIndex = kernel_exec(singleCell, operation, instruction, m_cellSize, m_cellsPerPU, m_numberOfPU, d_memory, d_output, h_output, h_cell);
	return (m_cellIndex != kCellNotFound);
}

bool snpDeviceImpl::read(uint32 *bitfield)
{
	if (m_cellIndex != kCellNotFound)
	{
		snpCudaSafeCall(cudaMemcpy(bitfield, d_memory + m_cellIndex, m_cellSize * sizeof(uint32), cudaMemcpyDeviceToHost));
		return true;
	}
	return false;
}

NS_SNP_END
