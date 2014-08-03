#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int32 kCellNotFound = -1;

//
// Returns absolute cell index in common memory array.
// Can affect performance significantly.
//
// cellIndex - index of cell within the single thread
// threadIndex - index of thread within the single block
// blockDim - total number of threads in the block
// blockIndex - index of block within the single grid
// gridDim - total number of blocks in the grid
//
__device__ uint32 snp_get_absolute_cell_index(uint32 cellIndex, uint32 threadIndex, uint32 blockDim, uint32 blockIndex, uint32 gridDim)
{
	return threadIndex + blockDim * (blockIndex + gridDim * cellIndex);
}

//
// Returns absolute index within output array for
// current thread (= PU).
//
__device__ __host__ uint32 snp_get_absolute_output_index(uint32 threadIndex, uint32 blockDim, uint32 blockIndex)
{
	return threadIndex + blockDim * blockIndex;
}

//
// Check 'cell' content which consists of 'cellSize' uint32 elements. Cell is
// selected when all its bits, which are marked by bits in 'mask' argument,
// are equal to bits in 'data' argument.
// Returns True if cell is selected.
//
__device__ bool snp_check_cell(const uint32 * const cell, const uint32 * const mask, const uint32 * const data, uint32 cellSize)
{
	for (uint32 index = 0; index < cellSize; index++)
	{
		// Project Properties: CUDA C/C++ / Device / Code Generation => "compute_11,sm_11" to use atomicAnd etc.
		if (snpCompareBits(cell[index], mask[index], data[index]) == false)
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
__device__ __host__ void snp_perform_cell(uint32 *cell, const uint32 * const mask, const uint32 * const data, uint32 cellSize, snp::snpOperation operation)
{
	// loop was moved into switch to reduce number of condition tests
	switch(operation)
	{
		case snp::snpAssign:	snpUpdateBits(snpUpdateBitsASSIGN,	cell[index], mask[index], data[index], cellSize);	break;
		case snp::snpNot:		snpUpdateBits(snpUpdateBitsNOT,		cell[index], mask[index], data[index], cellSize);	break;
		case snp::snpAnd:		snpUpdateBits(snpUpdateBitsAND,		cell[index], mask[index], data[index], cellSize);	break;
		case snp::snpOr:		snpUpdateBits(snpUpdateBitsOR,		cell[index], mask[index], data[index], cellSize);	break;
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
__global__ void snp_find_cell(const uint32 * const mask, const uint32 * const data, uint32 cellSize, uint32 cellsPerPU, uint32 *memory, int32 *output)
{
	uint32 outputIndex = snp_get_absolute_output_index(threadIdx.x, blockDim.x, blockIdx.x);
	for (uint32 cellIndex = 0; cellIndex < cellsPerPU; cellIndex++)
	{
		uint32 absoluteCellIndex = snp_get_absolute_cell_index(cellIndex, threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);
		uint32 *cell = &memory[absoluteCellIndex * cellSize];

		bool selected = snp_check_cell(cell, mask, data, cellSize);
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
	const uint32 * const dataMask, const uint32 * const dataData, uint32 cellSize, uint32 cellsPerPU, uint32 *memory, int32 *output)
{
	int32 result = kCellNotFound;
	for (int32 cellIndex = cellsPerPU - 1; cellIndex >= 0; cellIndex--)
	{
		uint32 absoluteCellIndex = snp_get_absolute_cell_index(cellIndex, threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);
		uint32 *cell = &memory[absoluteCellIndex * cellSize];

		bool selected = snp_check_cell(cell, addressMask, addressData, cellSize);
		if (selected == true)
		{
			result = cellIndex;
			snp_perform_cell(cell, dataMask, dataData, cellSize, operation);
		}
	}

	uint32 outputIndex = snp_get_absolute_output_index(threadIdx.x, blockDim.x, blockIdx.x);
	output[outputIndex] = result;
}

int32 kernel_exec(bool singleCell, snp::snpOperation operation, const uint32 * const instruction, uint32 cellSize,
	uint32 cellsPerPU, uint32 numberOfPU, uint32 *d_memory, int32 *d_output, int32 *h_output, uint32 *h_cell)
{
	// todo: get as much blocks as possible
	const uint32 numberOfBlocks = 16;

	// setup GPU launch configuration
	dim3 blockDim(numberOfPU / numberOfBlocks);	// number of threads (1 .. 1024)
	dim3 gridDim(numberOfBlocks);				// number of blocks (1 .. 65536)

	const uint32 * const addressMask	= instruction;
	const uint32 * const addressData	= instruction + 1 * cellSize;
	const uint32 * const dataMask		= instruction + 2 * cellSize;
	const uint32 * const dataData		= instruction + 3 * cellSize;

	// asynchronously runnung kernel on GPU
	if (singleCell == true)
	{
		snp_find_cell<<<gridDim, blockDim>>>(addressMask, addressData, cellSize, cellsPerPU, d_memory, d_output);
	}
	else
	{
		//snp_perform_instruction<<<blockDim, threadDim>>> (memory, output, cellsPerPU, numberOfPU, instruction);
	}

	// every thread (=PU) did write into 'output' array
	// the index of the selected cell within the PU (thread)

	// copy output array from GPU to CPU
	cudaMemcpy(h_output, d_output, numberOfPU * sizeof(int32), cudaMemcpyDeviceToHost);

	// analyze all selected cells to find the first one
	int32 absoluteCellIndex = kCellNotFound;
	for (uint32 puIndex = 0; puIndex < numberOfPU; puIndex++)
	{
		int32 cellIndex = h_output[puIndex];
		if (cellIndex != kCellNotFound)
		{
			absoluteCellIndex = puIndex + numberOfPU * cellIndex;
			break;
		}
	}

	if (singleCell == true && absoluteCellIndex != kCellNotFound)
	{
		// deferred update for the first selected cell

		// read selected cell from GPU
		cudaMemcpy(h_cell, d_memory + absoluteCellIndex * cellSize, cellSize * sizeof(uint32), cudaMemcpyDeviceToHost);

		// perform instruction to this cell
		snp_perform_cell(h_cell, dataMask, dataData, cellSize, operation);

		// write data back to GPU
		cudaMemcpy(d_memory + absoluteCellIndex * cellSize, h_cell, cellSize * sizeof(uint32), cudaMemcpyHostToDevice);
	}

	return absoluteCellIndex;
}

//
////// ядро CUDA, выполняется на GPU
////__global__ void addKernel(int *inputArray, const unsigned functionCode, int *outputArray, const unsigned inputSizeInt)
////{
////	// позиция = индекс блока * число нитей + номер нити (в примере, который обсуждался, - blockIdx.x = const = 0, threadIdx.x меняется от 0 до 511 в зависимости от номера нити)
////    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
////	int result = -1;
////
////	// в соотв. с кодом выполняем то или иное действие
////	switch( functionCode ) {
////		case 0:
////			for ( int i = pos; i < inputSizeInt; i += blockDim.x * gridDim.x ) {
////				// перебираем все элементы для данной нити с шагом blockDim.x * gridDim.x (число нитей * число блоков, в примере, который обсуждался, - blockDim.x = 512, gridDim.x = 1)
////
////				// действия, которые вам нужны
////				unsigned value = inputArray[i] + 1;
////
////				// условие, по которому решили, что действие завершено
////				bool wasFound = value == ( inputArray[i] + 1 );
////				if ( wasFound ) {
////					// сохраняем первый найденный индекс для данной нити
////					result = i;
////					break;
////				}
////			}
////			break;
////		// ...
////	}
////
////	// гарантируем, что все нити в рамках данного блока завершили свою работу
////	__syncthreads();
////
////	// сохраняем найденные индексы
////	outputArray[pos] = result;
////}
//
////int main()
////{
////	cudaError_t ret_cuda;
////	// инициализация выбранной видеокарты
////	CHECK_CUDA( cudaSetDevice(0) );
////
////	int *inputArray = new int[1024];
////	int *outputArray = new int[512];
////
////	for (int i = 0; i < 1024; i++)
////		inputArray[i] = i;
////	
////	// выделение памяти на gpu
////	int *d_inputarray, *d_outputarray;
////	check_cuda( cudamalloc((void**)&d_inputarray,  1024 * sizeof(int)) );
////	check_cuda( cudamalloc((void**)&d_outputarray,  512 * sizeof(int)) );
////
////	// копирование данных на gpu
////	check_cuda( cudamemcpy(d_inputarray,   inputarray, 1024 * sizeof(int), cudamemcpyhosttodevice) );
////
////	// настрока конфигурации запуска
////	dim3 threadDim(512);	// число нитей ( 1 ... 1024 ). blockDim.x в ядре теперь равен 512
////	dim3 blockDim(1);		// число блоков. gridDim.x в ядре теперь равен 1
////
////	// асинхронный запуск ядра на GPU
////	addKernel<<<blockDim, threadDim>>> ( d_inputArray, 0, d_outputArray, 1024 );
////
////	// проверка ошибок запуска
////	CHECK_CUDA( cudaGetLastError() );
////
////	// копирование данных из GPU
////	CHECK_CUDA( cudaMemcpy(outputArray, d_outputArray,  512 * sizeof(int), cudaMemcpyDeviceToHost) );
////
////	// в outputArray располагаются искомые данные
////	for (int i = 0; i < 512; i++) {
////		if ( outputArray[i] != i ) {
////			fprintf(stderr, "Pos = %d failed\n");
////			return -1;
////		}
////	}
////
////	CHECK_CUDA( cudaFree(  d_inputArray ) );
////	CHECK_CUDA( cudaFree( d_outputArray ) );
////	delete inputArray;
////	delete outputArray;
////
////	printf("Success\n");
////
////    return 0;
////}
