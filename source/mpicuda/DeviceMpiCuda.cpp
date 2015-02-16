#include <snp/Device.h>

#include "network/ProtocolHandler.h"
#include "network/SocketAcceptor.h"
#include "network/DeviceGlue.h"

NS_SNP_BEGIN

#ifdef WIN32

static class CWSAInitializer
{
public:
    CWSAInitializer()
    {
        WORD wVersionRequested = MAKEWORD(2, 2);
        WSADATA wsaData;

        int iError = WSAStartup(0x0202, &wsaData);
        if (iError != 0) {
            // Tell the user that we could not find a usable Winsock DLL.
            LOG_MESSAGE(1, "WSAStartup failed with error: %d\n", iError);
        }
    }

    ~CWSAInitializer()
    {
        WSACleanup();
    }

} s_oWSAInitializer;

#endif

static CProtocolHandler *s_pHandler = nullptr;

bool CDevice::Init(uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU)
{
    if (!uiCellSize) return false;

    m_uiCellSize = uiCellSize;
    m_uiCellsPerPU = uiCellsPerPU;
    m_uiNumberOfPU = uiNumberOfPU;

    // 1. read config from file (can be hardcoded for now) and run worker process on
    // the cluster host machine via ssh using parameters from the config
    // ...

    // connect to the worker process
    LOG_MESSAGE(1, "[snp-server] Connection...");
    
    CSocketAcceptor oSocketAcceptor;
    oSocketAcceptor.AcceptConnections();

    LOG_MESSAGE(1, "[snp-server] Connection established...");
    int iSocketFD = oSocketAcceptor.GetAcceptedSocket();

    assert(!s_pHandler);
    s_pHandler = new CProtocolHandler(iSocketFD);

    // send 'startup' command to the worker
    LOG_MESSAGE(1, "[snp-server] Starting system [%dx%dx%d]...", m_uiCellSize, m_uiCellsPerPU, m_uiNumberOfPU);
    bool bResult = glue::Startup(s_pHandler, uiCellSize, m_uiCellsPerPU, m_uiNumberOfPU);

    LOG_MESSAGE(1, "[snp-server] Final configuration [%dx%dx%d].", m_uiCellSize, m_uiCellsPerPU, m_uiNumberOfPU);
    return bResult;
}

void CDevice::Deinit()
{
    // send 'shutdown' command to the worker (cluster process will be stopped)
    LOG_MESSAGE(1, "[test-server] Shutting down...");
    glue::Shutdown(s_pHandler);

    // cleanup connection variables
    LOG_MESSAGE(1, "[snp-server] Close connection.");
    delete s_pHandler;
    s_pHandler = nullptr;

    // TODO: download log files from the cluster to local machine
    // ...
}

bool CDevice::Exec(bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction)
{
    // send 'exec' command to the worker
    return glue::Exec(s_pHandler, bSingleCell, eOperation, pInstruction, m_uiCellSize * 4);
}

bool CDevice::Read(uint32 *pBitfield)
{
    // 1. send 'read' command to the worker
    return glue::Read(s_pHandler, pBitfield);
}

NS_SNP_END
