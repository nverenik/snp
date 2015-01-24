#include <snp/snpDevice.h>
extern "C" const int32 kCellNotFound;

#include "snpBackendConfig.h"

NS_SNP_BEGIN

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
{
	m_exists = true;
}

snpDeviceImpl::~snpDeviceImpl()
{
	deinit();
	m_exists = false;
}

NS_SNP_END
