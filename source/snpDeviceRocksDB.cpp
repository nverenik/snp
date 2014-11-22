#include <snp-cuda\snpDevice.h>

#if (SNP_TARGET_BACKEND == SNP_BACKEND_ROCKS_DB)

extern "C" const int32 kCellNotFound = -1;

NS_SNP_BEGIN

bool snpDeviceImpl::init(uint16 cellSize, uint32 cellsPerPU, uint32 numberOfPU)
{
	return false;
}

void snpDeviceImpl::deinit()
{
}

bool snpDeviceImpl::exec(bool singleCell, snpOperation operation, const uint32 * const instruction)
{
	return false;
}

bool snpDeviceImpl::read(uint32 *bitfield)
{
	return false;
}

void snpDeviceImpl::dump()
{
}

NS_SNP_END

#endif //(SNP_TARGET_BACKEND == SNP_BACKEND_ROCKS_DB)
