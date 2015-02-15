#include <snp/Device.h>
extern "C" const int32 kCellNotFound;

NS_SNP_BEGIN

bool CDevice::m_bExists = false;

CDevice * CDevice::Create(uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU)
{
    CDevice *device = (m_bExists != true) ? new CDevice() : nullptr;
    if (device != nullptr && device->Init(uiCellSize, uiCellsPerPU, uiNumberOfPU) != true)
    {
        delete device;
        device = nullptr;
    }
    return device;
}

CDevice::CDevice()
    : m_uiCellSize(0)
    , m_uiCellsPerPU(0)
    , m_uiNumberOfPU(0)
{
    m_bExists = true;
}

CDevice::~CDevice()
{
    Deinit();
    m_bExists = false;
}

NS_SNP_END
