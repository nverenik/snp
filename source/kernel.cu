#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int32 kCellNotFound = -1;

//
// Returns absolute cell index in common memory array.
// Can affect performance significantly.
//
__device__ __host__ uint32 snp_get_absolute_cell_index(uint32 cellIndex, uint32 threadDim, uint32 threadIndex, uint32 blockDim, uint32 blockIndex)
{
	// TODO: should we precalculate blockDim * threadDim value?
	// the number of operations won't be changed
	return threadIndex + blockDim * (cellIndex + threadDim * blockIndex);
}

//
// Returns absolute index within output array for
// current thread (= PU).
//
__device__ uint32 snp_get_absolute_output_index(uint32 threadIndex, uint32 blockDim, uint32 blockIndex)
{
	return threadIndex + blockDim * blockIndex;
}

//
// Check 'cell' content which consists of 'cellSize' uint32 elements. Cell is
// selected when all its bits, which are marked by bits in 'mask' argument,
// are equal to bits in 'data' argument.
// Returns True if cell is selected.
//
__device__ bool snp_check_cell(const uint32 * const cell, const uint32 * const mask, const uint32 * const data, uint32 cellDim)
{
	for (uint32 index = 0; index < cellDim; index++)
	{
		if (snpCompareBits(cell[index], mask[index], data[index]) != true)
		{
			return false;
		}
	}
	return true;
}

//
// Perform basic bit operation on input 'cell' only for bits selected by marked bits in
// the 'mask' argument.
//
// TODO: replace with template method to avoid switch
//
__device__ __host__ void snp_perform_cell(uint32 *cell, const uint32 * const mask, const uint32 * const data, uint32 cellDim, snp::snpOperation operation)
{
	// loop was moved into switch to reduce number of condition tests
	switch(operation)
	{
		case snp::snpAssign:	snpUpdateBits(snpUpdateBitsASSIGN,	cell[index], mask[index], data[index], cellDim);	break;
		case snp::snpNot:		snpUpdateBits(snpUpdateBitsNOT,		cell[index], mask[index], data[index], cellDim);	break;
		case snp::snpAnd:		snpUpdateBits(snpUpdateBitsAND,		cell[index], mask[index], data[index], cellDim);	break;
		case snp::snpOr:		snpUpdateBits(snpUpdateBitsOR,		cell[index], mask[index], data[index], cellDim);	break;
		default: break;
	}
}

//
// Used in case of 'single cell' instruction as we don't need to find all selected cells.
// Doesn't change anything in memory.
//
// Returns relative to PU cell index of first selected cell (writes is into output array
// where each element is corresponds to single thread = PU).
//
__global__ void snp_find_cell(const uint32 * const mask, const uint32 * const data, uint32 cellDim, uint32 threadDim, uint32 *memory, int32 *output)
{
	uint32 outputIndex = snp_get_absolute_output_index(threadIdx.x, blockDim.x, blockIdx.x);
	for (uint32 cellIndex = 0; cellIndex < threadDim; cellIndex++)
	{
		uint32 absoluteCellIndex = snp_get_absolute_cell_index(cellIndex, threadDim, threadIdx.x, blockDim.x, blockIdx.x);
		uint32 *cell = &memory[absoluteCellIndex * cellDim];

		bool selected = snp_check_cell(cell, mask, data, cellDim);
		if (selected == true)
		{
			output[outputIndex] = cellIndex;
			return;
		}
	}
	output[outputIndex] = kCellNotFound;
}

//
// Used in case of 'multiple cell' instruction so we can perform instruction right now.
// Returns relative to PU cell index of first selected cell (writes is into output array
// where each element is corresponds to single thread = PU).
//
__global__ void snp_perform_instruction(snp::snpOperation operation, const uint32 * const addressMask, const uint32 * const addressData,
	const uint32 * const dataMask, const uint32 * const dataData, uint32 cellDim, uint32 threadDim, uint32 *memory, int32 *output)
{
	int32 result = kCellNotFound;
	for (int32 cellIndex = threadDim - 1; cellIndex >= 0; cellIndex--)
	{
		uint32 absoluteCellIndex = snp_get_absolute_cell_index(cellIndex, threadDim, threadIdx.x, blockDim.x, blockIdx.x);
		uint32 *cell = &memory[absoluteCellIndex * cellDim];

		bool selected = snp_check_cell(cell, addressMask, addressData, cellDim);
		if (selected == true)
		{
			result = cellIndex;
			snp_perform_cell(cell, dataMask, dataData, cellDim, operation);
		}
	}

	uint32 outputIndex = snp_get_absolute_output_index(threadIdx.x, blockDim.x, blockIdx.x);
	output[outputIndex] = result;
}

int32 kernel_exec(bool singleCell, snp::snpOperation operation, const uint32 * const instruction, uint32 cellSize,
	uint32 cellsPerPU, uint32 numberOfPU, uint32 *d_memory, uint32 *d_instruction, int32 *d_output, int32 *h_output, uint32 *h_cell)
{
	// setup GPU launch configuration
	// TODO: get as much blocks as possible
	const uint32 gridDim = 1;							// number of blocks (1 .. 65536) within grid
	const uint32 blockDim = numberOfPU / gridDim;		// number of threads (1 .. 1024) within block
	const uint32 &threadDim = cellsPerPU;				// number of cells within thread
	const uint32 &cellDim = cellSize;					// number of uint32 within cell

	// copy instruction from CPU memory to global GPU memory
	cudaMemcpy(d_instruction, instruction, 4 * cellDim * sizeof(uint32), cudaMemcpyHostToDevice);

	// prepare meaningful bitfield names (for CPU and GPU)
	const uint32 * const dataMask		= instruction + 2 * cellSize;
	const uint32 * const dataData		= instruction + 3 * cellSize;

	const uint32 * const d_addressMask	= d_instruction;
	const uint32 * const d_addressData	= d_instruction + 1 * cellSize;
	const uint32 * const d_dataMask		= d_instruction + 2 * cellSize;
	const uint32 * const d_dataData		= d_instruction + 3 * cellSize;

	// asynchronously runnung kernel on GPU
	if (singleCell == true)
	{
		snp_find_cell<<<dim3(gridDim), dim3(blockDim)>>>(d_addressMask, d_addressData, cellDim, threadDim, d_memory, d_output);
	}
	else
	{
		snp_perform_instruction<<<dim3(gridDim), dim3(blockDim)>>>(operation, d_addressMask, d_addressData, d_dataMask, d_dataData, cellDim, threadDim, d_memory, d_output);
	}

	//cudaDeviceSynchronize();

	// test after kerner finished
	//cudaError_t ret_cuda = cudaGetLastError();

	// every thread (=PU) did write into 'output' array
	// the index of the selected cell within the PU (thread)

	// copy output array from GPU to CPU
	cudaMemcpy(h_output, d_output, numberOfPU * sizeof(int32), cudaMemcpyDeviceToHost);

	//cudaError_t ret_cuda = cudaGetLastError();
	//if (ret_cuda != cudaSuccess) {
	//	printf("%s\n", cudaGetErrorString(ret_cuda));
	//}

	// analyze all selected cells to find the first one
	int32 absoluteCellIndex = kCellNotFound;
	for (uint32 puIndex = 0; puIndex < numberOfPU; puIndex++)
	{
		int32 cellIndex = h_output[puIndex];
		if (cellIndex != kCellNotFound)
		{
			uint32 threadIndex = puIndex % blockDim;
			uint32 blockIndex = puIndex / blockDim;
			absoluteCellIndex = snp_get_absolute_cell_index(cellIndex, threadDim, threadIndex, blockDim, blockIndex);
			break;
		}
	}

	if (singleCell == true && absoluteCellIndex != kCellNotFound)
	{
		// deferred update for the first selected cell

		// read selected cell from GPU
		cudaMemcpy(h_cell, d_memory + absoluteCellIndex * cellDim, cellDim * sizeof(uint32), cudaMemcpyDeviceToHost);

		// perform instruction to this cell
		snp_perform_cell(h_cell, dataMask, dataData, cellDim, operation);

		// write data back to GPU
		cudaMemcpy(d_memory + absoluteCellIndex * cellDim, h_cell, cellDim * sizeof(uint32), cudaMemcpyHostToDevice);
	}

	return absoluteCellIndex;
}
