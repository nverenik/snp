#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <snp/snpMacros.h>
#include <snp/snpOperation.h>

extern "C" const int32 kCellNotFound;

extern "C" int32 kernel_exec(
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
    uint32  *h_pCell
);

#endif //__KERNEL_H__
