#ifndef __WORKER_H__
#define __WORKER_H__

#include <mpi.h>
#include <cuda_runtime.h>

#include <vector>
#include <snp/snpMacros.h>

NS_SNP_BEGIN

class CWorker
{
public:
    static bool IsCUDASupported();
    CWorker(MPI_Comm oCommunicator, int32 iHostRank);

    // collect information about mpi nodes
    bool Init();
    bool PrintSystemInfo() const;

    void RunLoop();

    inline MPI_Comm GetCommunicator() const { return m_oCommunicator; }
    inline int32    GetGroupSize() const {return m_iGroupSize; }
    inline int32    GetHostRank() const { return m_iHostRank; }
    inline int32    GetRank() const { return m_iRank; }
    inline bool     IsHost() const { return m_bHost; }

private:
    typedef cudaDeviceProp snpDeviceInfo;
    typedef std::vector<snpDeviceInfo> snpNodeInfo;
    typedef std::vector<snpNodeInfo> snpSystemInfo;

    enum tCommand
    {
        tCommand_Idle = 0,
        //tCommand_SystemInfo,
        tCommand_Startup,
        tCommand_Shutdown,
        tCommand_Exec,
        tCommand_Read,
    };

    bool Startup(/*int32 iRank, const SystemInfo &roSystemInfo, uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU*/);
    bool Shutdown();
    bool Exec(/*int32 iRank, bool bSingleCell, snpOperation eOperation, const uint32 * const pInstruction*/);
    bool Read(/*int32 iRank, uint32 *pBitfield*/);

    // MPI constants
    const MPI_Comm  m_oCommunicator;
    const int32     m_iHostRank;

    // general MPI process info
    char            m_pszProcessorName[MPI_MAX_PROCESSOR_NAME];
    int32           m_iGroupSize;
    int32           m_iRank;
    bool            m_bHost;
    snpNodeInfo     m_oNodeInfo;

    // host specific data
    snpSystemInfo   m_oSystemInfo;
};

NS_SNP_END

#endif //__WORKER_H__
