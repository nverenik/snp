#include "Kernel.h"

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
__device__ __host__ void snp_perform_cell(uint32 *cell, const uint32 * const mask, const uint32 * const data, uint32 cellDim, snp::tOperation operation)
{
    // loop was moved into switch to reduce number of condition tests
    switch(operation)
    {
        case snp::tOperation_Assign:    snpUpdateBits(snpUpdateBitsASSIGN,  cell[index], mask[index], data[index], cellDim);    break;
        case snp::tOperation_Not:       snpUpdateBits(snpUpdateBitsNOT,     cell[index], mask[index], data[index], cellDim);    break;
        case snp::tOperation_And:       snpUpdateBits(snpUpdateBitsAND,     cell[index], mask[index], data[index], cellDim);    break;
        case snp::tOperation_Or:        snpUpdateBits(snpUpdateBitsOR,      cell[index], mask[index], data[index], cellDim);    break;
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
__global__ void snp_perform_instruction(snp::tOperation operation, const uint32 * const addressMask, const uint32 * const addressData,
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

int32 kernel_exec(
    bool bSingleCell,
    snp::tOperation eOperation,
    const uint32 * const pInstruction,
    uint32  uiCellDim,
    uint32  uiThreadDim,
    uint32  uiBlockDim,
    uint32  uiGridDim,
    uint32  *d_pMemory,
    uint32  *d_pInstruction,
    int32   *d_pOutput,
    int32   *h_pOutput,
    uint32  *h_pCell)
{
    const uint32 uiNumberOfPU = uiGridDim * uiBlockDim;

    // copy instruction from CPU memory to global GPU memory
    cudaMemcpy(d_pInstruction, pInstruction, 4 * uiCellDim * sizeof(uint32), cudaMemcpyHostToDevice);

    // prepare meaningful bitfield names (for CPU and GPU)
    const uint32 * const dataMask       = pInstruction + 2 * uiCellDim;
    const uint32 * const dataData       = pInstruction + 3 * uiCellDim;

    const uint32 * const d_addressMask  = d_pInstruction;
    const uint32 * const d_addressData  = d_pInstruction + 1 * uiCellDim;
    const uint32 * const d_dataMask     = d_pInstruction + 2 * uiCellDim;
    const uint32 * const d_dataData     = d_pInstruction + 3 * uiCellDim;

    // asynchronously runnung kernel on GPU
    if (bSingleCell == true)
    {
        snp_find_cell<<<dim3(uiGridDim), dim3(uiBlockDim)>>>(d_addressMask, d_addressData, uiCellDim, uiThreadDim, d_pMemory, d_pOutput);
    }
    else
    {
        snp_perform_instruction<<<dim3(uiGridDim), dim3(uiBlockDim)>>>(eOperation, d_addressMask, d_addressData, d_dataMask, d_dataData, uiCellDim, uiThreadDim, d_pMemory, d_pOutput);
    }

    //cudaDeviceSynchronize();

    // test after kerner finished
    //cudaError_t ret_cuda = cudaGetLastError();

    // every thread (=PU) did write into 'output' array
    // the index of the selected cell within the PU (thread)

    // copy output array from GPU to CPU
    cudaMemcpy(h_pOutput, d_pOutput, uiNumberOfPU * sizeof(int32), cudaMemcpyDeviceToHost);

    //cudaError_t ret_cuda = cudaGetLastError();
    //if (ret_cuda != cudaSuccess) {
    //    printf("%s\n", cudaGetErrorString(ret_cuda));
    //}

    // analyze all selected cells to find the first one
    int32 iAbsoluteCellIndex = kCellNotFound;
    for (uint32 uiPUIndex = 0; uiPUIndex < uiNumberOfPU; uiPUIndex++)
    {
        int32 iCellIndex = h_pOutput[uiPUIndex];
        if (iCellIndex != kCellNotFound)
        {
            uint32 uiThreadIndex = uiPUIndex % uiBlockDim;
            uint32 uiBlockIndex = uiPUIndex / uiBlockDim;
            iAbsoluteCellIndex = snp_get_absolute_cell_index(iCellIndex, uiThreadDim, uiThreadIndex, uiBlockDim, uiBlockIndex);
            break;
        }
    }

    if (bSingleCell == true && iAbsoluteCellIndex != kCellNotFound)
    {
        // deferred update for the first selected cell

        // read selected cell from GPU
        cudaMemcpy(h_pCell, d_pMemory + iAbsoluteCellIndex * uiCellDim, uiCellDim * sizeof(uint32), cudaMemcpyDeviceToHost);

        // perform instruction to this cell
        snp_perform_cell(h_pCell, dataMask, dataData, uiCellDim, eOperation);

        // write data back to GPU
        cudaMemcpy(d_pMemory + iAbsoluteCellIndex * uiCellDim, h_pCell, uiCellDim * sizeof(uint32), cudaMemcpyHostToDevice);
    }

    return iAbsoluteCellIndex;
}
