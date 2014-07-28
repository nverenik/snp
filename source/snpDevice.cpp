#include <snp-cuda\snpDevice.h>

NS_SNP_BEGIN

bool snpDeviceImpl::m_exists = false;

snpDeviceImpl * snpDeviceImpl::create(uint16 cellSize, uint32 cellsPerPU, uint32 numberOfPU)
{
	return (m_exists != true) ? new snpDeviceImpl() : nullptr;
}

snpDeviceImpl::snpDeviceImpl()
	: m_cellsPerPU(0)
	, m_numberOfPU(0)
{
	m_exists = true;
}

snpDeviceImpl::~snpDeviceImpl()
{
	m_exists = false;
}

NS_SNP_END
