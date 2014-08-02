//#include "kernel.h"
//#include "snpMacros.h"
//USING_NS_SNP;
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <cstring>
//
//const int kCellNotFound = -1;
//
////
//// Returns absolute cell index in common memory array.
//// Can affect performance significantly.
////
//__device__ __host__ uint32 snp_get_absolute_cell_index(uint32 threadIndex, uint32 numberOfPU, uint32 cellIndex)
//{
//	return threadIndex + numberOfPU * cellIndex;
//}
//
////
//// Check input cell content which consists of 'snpSizeOfCell' int elements. Cell is
//// selected when all its bits, which are selected by raised bits in mask argument,
//// are equal to bits in data argument.
//// Returns True if cell is selected.
////
//__device__ bool snp_check_cell(snpBitfield &cell, snpBitfield &mask, snpBitfield &data)
//{
//	for (uint32 index = 0; index < snpSizeOfCell; index++)
//	{
//		// Project Properties: CUDA C/C++ / Device / Code Generation => "compute_11,sm_11" to use atomicAnd etc.
//		if (snpCompareBits(cell.bitfield[index], mask.bitfield[index], data.bitfield[index]) == false)
//			return false;
//	}
//	return true;
//}
//
////
//// Perform basic bit operation on input cell only for bits selected by raised bits in
//// mask argument.
////
//__device__ __host__ void snp_perform_cell(snpBitfield &cell, const snpBitfield &mask, const snpBitfield &data, snpOperation operation)
//{
//	for (uint32 index = 0; index < snpSizeOfCell; index++)
//	{
//		switch (operation)
//		{
//			case snpAssign:	snpUpdateBitsASSIGN(cell.bitfield[index], mask.bitfield[index], data.bitfield[index]);	break;
//			case snpNot:	snpUpdateBitsNOT(cell.bitfield[index], mask.bitfield[index]);							break;
//			case snpAnd:	snpUpdateBitsAND(cell.bitfield[index], mask.bitfield[index], data.bitfield[index]);		break;
//			case snpOr:		snpUpdateBitsOR(cell.bitfield[index], mask.bitfield[index], data.bitfield[index]);		break;
//		}
//	}
//}
//
////
//// Used in case of 'single cell' instruction so we don't need to find all selected cells.
//// Don't change anything in memory.
//// Returns relative to PU cell index of first selected cell (writes is into output array
//// where each element is corresponds to single thread = PU).
////
//__global__ void snp_find_cell(snpBitfield *memory, int32 *output, uint32 cellsPerPU, uint32 numberOfPU, snpInstruction instruction)
//{
//	for (int32 cellIndex = 0; cellIndex < cellsPerPU; cellIndex++)
//	{
//		int32 absoluteCellIndex = snp_get_absolute_cell_index(threadIdx.x, numberOfPU, cellIndex);
//		snpBitfield &cellBitfield = memory[absoluteCellIndex];
//
//		bool selected = snp_check_cell(cellBitfield, instruction.addressMask, instruction.addressData);
//		if (selected == true)
//		{
//			output[threadIdx.x] = cellIndex;
//			return;
//		}
//	}
//	output[threadIdx.x] = kCellNotFound;
//}
//
////
//// Used in case of 'multiple cell' instruction so we can perform instruction right now.
//// Returns relative to PU cell index of first selected cell (writes is into output array
//// where each element is corresponds to single thread = PU).
////
//__global__ void snp_perform_instruction(snpBitfield *memory, int32 *output, uint32 cellsPerPU, uint32 numberOfPU, snpInstruction instruction)
//{
//	output[threadIdx.x] = kCellNotFound;
//	// threadIdx.x = [0..(threadDim-1)]
//	// blockIdx.x  = [0..(blockDim-1)]
//	// blockDim.x = const = threadDim
//	// gridDim.x  = const = blockDim (макс.размер при конфигурации)
//	for (int32 cellIndex = cellsPerPU - 1; cellIndex >= 0; cellIndex--)
//	{
//		int32 absoluteCellIndex = snp_get_absolute_cell_index(threadIdx.x, numberOfPU, cellIndex);
//		snpBitfield &cellBitfield = memory[absoluteCellIndex];
//
//		bool selected = snp_check_cell(cellBitfield, instruction.addressMask, instruction.addressData);
//		if (selected == true)
//		{
//			output[threadIdx.x] = cellIndex;
//			snp_perform_cell(cellBitfield, instruction.dataMask, instruction.dataData, instruction.operation);
//		}
//	}
//}
//
//int kernel_exec(snpBitfield *memory, int32 *output, uint32 cellsPerPU, uint32 numberOfPU, const snpInstruction &instruction)
//{
//	// setup GPU launch configuration
//	dim3 threadDim(numberOfPU);	// число нитей ( 1 ... 1024 ).
//	dim3 blockDim(1);		// число блоков (1..65535).
//
//	// asynchronously runnung kernel on GPU
//	if (instruction.singleCell)
//	{
//		snp_find_cell<<<blockDim, threadDim>>>(memory, output, cellsPerPU, numberOfPU, instruction);
//	}
//	else
//	{
//		snp_perform_instruction<<<blockDim, threadDim>>> (memory, output, cellsPerPU, numberOfPU, instruction);
//	}
//	/*
//	cudaError_t cudaMemGetInfo 	( 	size_t *  	free,
//									size_t *  	total	 
//	) 	
//	*/
//	// every PU wrote related index of selected cell into output array
//	// allocate memory on CPU
//	int32 *outputCPU = new int32[numberOfPU];
//	memset(outputCPU, 0, numberOfPU * sizeof(int32));
//	
//	// copy output array from GPU to CPU
//	cudaMemcpy(outputCPU, output, numberOfPU * sizeof(int32), cudaMemcpyDeviceToHost);
//
//	// analyze all selected cells to find the first one
//	int32 absoluteCellIndex = kCellNotFound;
//	for (uint32 puIndex = 0; puIndex < numberOfPU; puIndex++)
//	{
//		int32 cellIndex = outputCPU[puIndex];
//		if (cellIndex != kCellNotFound)
//		{
//			absoluteCellIndex = snp_get_absolute_cell_index(puIndex, numberOfPU, cellIndex);
//			break;
//		}
//	}
//
//	delete outputCPU;
//
//	if (instruction.singleCell && absoluteCellIndex != kCellNotFound)
//	{
//		// deferred update for only first selected cell
//
//		// read selected cell from GPU
//		snpBitfield cellBitfield;
//		cudaMemcpy(&cellBitfield, memory + absoluteCellIndex, sizeof(snpBitfield), cudaMemcpyDeviceToHost);
//
//		// perform instruction
//		snp_perform_cell(cellBitfield, instruction.dataMask, instruction.dataData, instruction.operation);
//
//		// write data back to GPU
//		cudaMemcpy(memory + absoluteCellIndex, &cellBitfield, sizeof(snpBitfield), cudaMemcpyHostToDevice);
//	}
//
//	return absoluteCellIndex;
//}
//
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
