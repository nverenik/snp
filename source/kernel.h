#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <snp-cuda\snpMacros.h>
#include <snp-cuda\snpOperation.h>

extern "C" const int32 kCellNotFound;

extern "C" int32 kernel_exec(
	bool singleCell,
	snp::snpOperation operation,
	const uint32 * const instruction,
	uint32 cellSize,
	uint32 cellsPerPU,
	uint32 numberOfPU,
	uint32 *d_memory,
	uint32 *d_instruction,
	int32 *d_output,
	int32 *h_output,
	uint32 *h_cell
);

#endif //__KERNEL_H__
