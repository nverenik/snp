#ifndef __WORKER_H__
#define __WORKER_H__

#include <mpi.h>
#include <cuda_runtime.h>

#include <vector>
#include <snp/snp.h>

class CProtocolHandler;

NS_SNP_BEGIN

class CWorker
{
public:
    static bool IsCUDASupported();
    CWorker(MPI_Comm oCommunicator, int32 iHostRank);

    // collect information about mpi nodes
    bool Init();
    bool PrintSystemInfo() const;
    void RunLoop(CProtocolHandler *pHandler);

    inline MPI_Comm GetCommunicator() const { return m_oCommunicator; }
    inline int32    GetGroupSize() const {return m_iGroupSize; }
    inline int32    GetHostRank() const { return m_iHostRank; }
    inline int32    GetRank() const { return m_iRank; }
    inline bool     IsHost() const { return m_bHost; }

private:
    typedef cudaDeviceProp tDeviceInfo;
    typedef std::vector<tDeviceInfo> tNodeInfo;
    typedef std::vector<tNodeInfo> tSystemInfo;

    struct tDeviceConfig;
    typedef std::vector<tDeviceConfig> tNodeConfig;
    typedef std::vector<tNodeConfig> tSystemConfig;

    enum tCommand
    {
        tCommand_Idle = 0,
        //tCommand_SystemInfo,
        tCommand_Startup,
        tCommand_Shutdown,
        tCommand_Exec,
        tCommand_Read,
    };

    struct tDeviceConfig
    {
        uint32    m_uiGridDim;    // number of blocks (1 .. 65536) within grid
        uint32    m_uiBlockDim;    // number of threads (1 .. 1024) within block
        uint32    m_uiThreadDim;    // number of cells within thread
        uint32    m_uiCellDim;    // number of uint32 within cell
    };

    void Tick();

    bool Startup(uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU);
    bool Shutdown();
    bool Exec(bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction);
    bool Read(uint32 *pBitfield);

    bool ExecSequential(tOperation eOperation, const uint32 * const pInstruction);
    bool ExecParallel(tOperation eOperation, const uint32 * const pInstruction);
    bool ExecImpl(bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction);

    // MPI constants
    const MPI_Comm  m_oCommunicator;
    const int32     m_iHostRank;

    // host specific data
    tSystemInfo     m_oSystemInfo;

    // general MPI process info
    char        m_pszProcessorName[MPI_MAX_PROCESSOR_NAME];
    int32       m_iGroupSize;
    int32       m_iRank;
    bool        m_bHost;
    tNodeInfo   m_oNodeInfo;    // info about all available devices in the current node
    tNodeConfig m_oNodeConfig;  // configuration for each device in the current node
    bool        m_bShouldExit;

    // pointers to the memory allocated on devices
    std::vector<uint32 *>   d_aMemory;
    std::vector<uint32 *>   d_aInstruction;
    std::vector<int32 *>    d_aOutput;

    // pre-allocated buffers used when working with kernel
    std::vector<int32 *>    h_aOutput;
    std::vector<uint32 *>   h_aCell;

    // Result of ther last performed Exec() command is the index of device
    // and index of the cell inside its memory pointing to the first matched
    // cell during instruction
    int32   m_iNodeIndex;   // is valid only on host mpi node
    int32   m_iDeviceIndex;
    int32   m_iCellIndex;
};

NS_SNP_END

#endif //__WORKER_H__
