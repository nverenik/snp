#ifndef __DEVICE_H__
#define __DEVICE_H__

#include <snp/Macros.h>
#include <snp/ErrorCode.h>
#include <snp/Operation.h>

NS_SNP_BEGIN

class CDevice;

template<uint16 uiBitwidth>
class tmDevice
{
private:
    // TODO: change round to ceil method
    static const uint16 s_uiCellSize = static_cast<uint16>(uiBitwidth / (sizeof(uint32) * 8) + 0.5f);

public:
    struct tBitfield
    {
        uint32 _raw[s_uiCellSize];
    };

    typedef union
    {
        struct
        {
            uint32 _raw[s_uiCellSize * 4];
        };

        struct
        {
            tBitfield   m_oSearchMask;
            tBitfield   m_oSearchTag;
            tBitfield   m_oWriteMask;
            tBitfield   m_oWriteData;
        };
    } tInstruction;

public:
    tmDevice();
    ~tmDevice();

    CErrorCode Configure(uint32 uiCellsPerPU, uint32 uiNumberOfPU);
    CErrorCode End();

    inline bool IsReady() const { return m_pDevice != nullptr; }

    static uint16 GetCellSize() { return s_uiCellSize; }
    inline uint32 GetCellsPerPU() const;
    inline uint32 GetNumberOfPU() const;

    // Execute instruction on device. Returns True if at least one cell activated.
    bool Exec(bool bSingleCell, tOperation eOperation, const tInstruction &roInstruction, CErrorCode *pError = nullptr);

    // Read data from the 1st cell which activated while the last instruction.
    // Returns False if no one cell is selected.
    bool Read(tBitfield &roBitfield, CErrorCode *pError = nullptr);

private:
    CDevice *m_pDevice;
};

class CDevice
{
private:
    static CDevice * Create(uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU);

    CDevice();
    ~CDevice();

    bool Init(uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU);
    void Deinit();

    bool Exec(bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction);
    bool Read(uint32 *pBitfield);

    inline uint32 GetCellsPerPU() const    { return m_uiCellsPerPU; }
    inline uint32 GetNumberOfPU() const    { return m_uiNumberOfPU; }

    uint32  m_uiCellSize;
    uint32  m_uiCellsPerPU;
    uint32  m_uiNumberOfPU;

    template<uint16 uiBitwidth>
    friend class tmDevice;

    static bool m_bExists;    
};

template<uint16 uiBitwidth>
tmDevice<uiBitwidth>::tmDevice()
    : m_pDevice(nullptr)
{
}

template<uint16 uiBitwidth>
tmDevice<uiBitwidth>::~tmDevice()
{
    if (m_pDevice != nullptr)
    {
        delete m_pDevice;
    }
}

template<uint16 uiBitwidth>
CErrorCode tmDevice<uiBitwidth>::Configure(uint32 uiCellsPerPU, uint32 uiNumberOfPU)
{
    if (m_pDevice != nullptr)
    {
        return CErrorCode::DEVICE_ALREADY_CONFIGURED;
    }

    m_pDevice = CDevice::Create(s_uiCellSize, uiCellsPerPU, uiNumberOfPU);
    return (m_pDevice != nullptr) ? CErrorCode::SUCCEEDED : CErrorCode::GPU_INIT_ERROR;
}

template<uint16 uiBitwidth>
CErrorCode tmDevice<uiBitwidth>::End()
{
    if (m_pDevice == nullptr)
    {
        return CErrorCode::DEVICE_NOT_CONFIGURED;
    }

    delete m_pDevice;
    m_pDevice = nullptr;

    return CErrorCode::SUCCEEDED;
}

template<uint16 uiBitwidth>
bool tmDevice<uiBitwidth>::Exec(bool bSingleCell, tOperation eOperation, const tInstruction &roInstruction, CErrorCode *pError/* = nullptr*/)
{
    if (m_pDevice != nullptr)
    {
        if (pError != nullptr)
            (*pError) = CErrorCode::SUCCEEDED;

        return m_pDevice->Exec(bSingleCell, eOperation, roInstruction._raw);
    }
    
    if (pError != nullptr)
        (*pError) = CErrorCode::DEVICE_NOT_CONFIGURED;

    return false;
}

template<uint16 uiBitwidth>
bool tmDevice<uiBitwidth>::Read(tBitfield &roBitfield, CErrorCode *pError/* = nullptr*/)
{
    if (m_pDevice != nullptr)
    {
        if (pError != nullptr)
            (*pError) = CErrorCode::SUCCEEDED;

        return m_pDevice->Read(roBitfield._raw);
    }
    
    if (pError != nullptr)
        (*pError) = CErrorCode::DEVICE_NOT_CONFIGURED;

    return false;
}

template<uint16 uiBitwidth>
uint32 tmDevice<uiBitwidth>::GetCellsPerPU() const
{
    return (m_pDevice != nullptr) ? m_pDevice->GetCellsPerPU() : 0;
}

template<uint16 uiBitwidth>
uint32 tmDevice<uiBitwidth>::GetNumberOfPU() const
{
    return (m_pDevice != nullptr) ? m_pDevice->GetNumberOfPU() : 0;
}

NS_SNP_END

#endif //__DEVICE_H__
