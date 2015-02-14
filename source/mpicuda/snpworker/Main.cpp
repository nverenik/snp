#include <mpi.h>

#include <stdio.h>
#include <memory>

#include <tclap/CmdLine.h>

#ifdef WIN32
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#endif // WIN32

#include "../network/DataTypes.h"
#include "../network/RenameMe.h"
#include "../network/ProtocolHandler.h"

#include "Worker.h"

struct Config;

static void OnExit();
static bool ProcessCommandLine(int32 argc, char* argv[], Config &roConfig);
static int32 Connect(const std::string &sHost, int iPort);

extern "C" void * ThreadServerF(void *pArg);

struct Config
{
    bool    m_bTestEnabled;
    bool    m_bLogSystemInfo;
};

//#define MPI_LOG(__format__, ...) printf("[%d:%s] "__format__"\n", s_iMpiRank, s_pszProcessorName, ##__VA_ARGS__)

int main(int argc, char* argv[])
{
    system("hostname");

    ::atexit(OnExit);
    MPI_Init(&argc, &argv);

    // TODO: replace communicator by group of nodes with suitable GPUs
    snp::CWorker oWorker(/*communicator handler*/MPI_COMM_WORLD, /*host process rank*/0);

    bool bExit = false;

    // host node parses command line
    Config oConfig;
    if (oWorker.IsHost() && !ProcessCommandLine(argc, argv, oConfig))
        bExit = true;

    // and finish all processes in case of error
    MPI_Bcast(&bExit, 1, MPI_BYTE, oWorker.GetHostRank(), oWorker.GetCommunicator());
    if (bExit) return 0;

    oWorker.Init();
    if (oWorker.IsHost() && oConfig.m_bLogSystemInfo)
    {
        oWorker.PrintSystemInfo();
        // todo: abort slave nodes as well
        return 0;
    }

    if (oWorker.IsHost() && oConfig.m_bTestEnabled)
    {
        // run separated thread which emulates main app
        pthread_t hThreadServer;
        if (pthread_create(&hThreadServer, NULL, ThreadServerF, NULL) != 0)
        {
            //LOG_MESSAGE( 1, "Error creating Server thread: %s", strerror(errno) );
            // todo: abort slave nodes as well
            return 0;
        }
    }

    // connect to the main application
    if (oWorker.IsHost())
    {
        int32 iSocketFD = Connect("127.0.0.1", 60666);
	    if (iSocketFD == -1)
        {
            // todo: abort slave nodes as well
            return 0;
        }

        CProtocolHandler oHanler(iSocketFD);
        oWorker.RunLoop(&oHanler);
    }
    else
    {
        oWorker.RunLoop(NULL);
    }

    // MPI_Finalize() is called using atexit callback
    return 0;
}

static void OnExit()
{
    MPI_Finalize();
}

static bool ProcessCommandLine(int argc, char* argv[], Config &roConfig)
{
    try
    {
        TCLAP::CmdLine oCommandLine("The worker executable is a part of software imitation model of the associative SNP "
            "(Semantic Network Processor, see http://github.com/nverenik/snp). It's responsible "
            "for connection with the main application and execution received commands on the "
            "computation cluster using MPI and NVidia CUDA frameworks.", ' ', "0.1.0");
        TCLAP::SwitchArg oTestSwitch("", "test", "Runs host worker in the test mode. It will generate a sequence of dummy "
            "commands and send them to the nodes.", oCommandLine, false);
        TCLAP::SwitchArg oInfoSwitch("", "info", "Displays detailed cluster information and exits.", oCommandLine, false);
        
        // Parse the argv array.
        oCommandLine.parse(argc, argv);
        roConfig.m_bTestEnabled = oTestSwitch.getValue();
        roConfig.m_bLogSystemInfo = oInfoSwitch.getValue();
    }
    catch (...)
    {
        return false;
    }
    return true;
}

static int32 Connect(const std::string &sHost, int32 iPort)
{
    //LOG_MESSAGE(3, "SocketConnector: Connecting to %s:%d", sHost.c_str(), iPort);

    uint64 iHost = inet_addr(sHost.c_str());
    if(iHost == INADDR_NONE)
    {
        struct hostent* pHostent = gethostbyname( sHost.c_str() );
        if(!pHostent)
        {
            //LOG_MESSAGE( 1, "gethostbyname error: %s", hstrerror(h_errno) );
            return -1;
        }
        iHost = *(unsigned long*)(pHostent->h_addr);
    }

    struct sockaddr_in oSockaddr;
    memset( &oSockaddr, 0, sizeof(oSockaddr) );
    oSockaddr.sin_family = AF_INET;
    oSockaddr.sin_port = htons(iPort);
    oSockaddr.sin_addr.s_addr = 
#ifdef WIN32
        iHost;
#else
        (in_addr_t)iHost;
#endif // WIN32

    int iNewSocketFD;
    if ((iNewSocketFD = socket(AF_INET, SOCK_STREAM, 0)) == -1)
    {
        LOG_MESSAGE(1, "socket error: %s", strerror(errno));
        return -1;
    }
    if (connect(iNewSocketFD, (struct sockaddr *) &oSockaddr, sizeof(oSockaddr)) == -1)
    {
        LOG_MESSAGE(1, "connect error: %s", strerror(errno));
        CloseSocket(iNewSocketFD);
        return -1;
    }
    else
    {
//#ifdef WIN32
//        // Set the socket I/O mode: In this case FIONBIO
//        // enables or disables the blocking mode for the 
//        // socket based on the numerical value of iMode.
//        // If iMode = 0, blocking is enabled; 
//        // If iMode != 0, non-blocking mode is enabled.
//        u_long iMode = 1;
//        ioctlsocket(iNewSocketFD, FIONBIO, &iMode);
//#else
//	    fcntl(iNewSocketFD, F_SETFL, O_NONBLOCK);
//#endif // WIN32
    }

    return iNewSocketFD;
}

//static bool Exec(int32 iRank, bool bSingleCell, snpOperation eOperation, const uint32 * const pInstruction)
//{
//    // execute kernel function for each device
//    s_iDeviceIndex = kCellNotFound;
//    for (uint32 iDeviceIndex = 0; iDeviceIndex < s_oNodeConfiguration.size(); iDeviceIndex++)
//    {
//        cudaError_t eErrorCode = cudaSetDevice(iDeviceIndex);
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        const DeviceConfiguration &roDeviceConfiguration = s_oNodeConfiguration[iDeviceIndex];
//        s_iCellIndex = kernel_exec(
//            bSingleCell,
//            eOperation,
//            pInstruction,
//            roDeviceConfiguration.m_uiCellDim,
//            roDeviceConfiguration.m_uiThreadDim,
//            roDeviceConfiguration.m_uiBlockDim,
//            roDeviceConfiguration.m_uiGridDim,
//            d_aMemory[iDeviceIndex],
//            d_aInstruction[iDeviceIndex],
//            d_aOutput[iDeviceIndex],
//            h_aOutput[iDeviceIndex],
//            h_aCell[iDeviceIndex]
//        );
//
//        if (s_iCellIndex != kCellNotFound)
//        {
//            s_iDeviceIndex = iDeviceIndex;
//            break;
//        }
//    }
//
//    // share with host just the fact that cell is found
//    // bool bFound = (s_iDeviceIndex != kCellNotFound && s_iCellIndex != kCellNotFound);
//    uint8 uiFound = (s_iDeviceIndex != kCellNotFound && s_iCellIndex != kCellNotFound) ? 1 : 0;
//
//    // prepare buffer to receive
//    int32 iGroupSize = 0;
//    MPI_Comm_size(MPI_COMM_WORLD, &iGroupSize);
//
//    // std::vector<bool> aFound;
//    std::vector<uint8> aFound;
//    if (iRank == s_iMpiHostRank)
//        aFound.resize(iGroupSize);
//
//    // share result
//    //MPI_Gather((void *)&bFound, 1, MPI_BYTE, (void *)(&aFound.front()), 1, MPI_BYTE, s_iMpiHostRank, MPI_COMM_WORLD);
//    MPI_Gather(&uiFound, 1, MPI_BYTE, &aFound.front(), 1, MPI_BYTE, s_iMpiHostRank, MPI_COMM_WORLD);
//
//    // find the first matched node
//    if (iRank == s_iMpiHostRank)
//    {
//        s_iNodeIndex = kCellNotFound;
//        for (uint32 iNodeIndex = 0; iNodeIndex < aFound.size(); iNodeIndex++)
//        {
//            if (aFound[iNodeIndex])
//            {
//                s_iNodeIndex = iNodeIndex;
//                break;
//            }
//        }
//        return (s_iNodeIndex != kCellNotFound);
//    }
//    return uiFound;
//}
//
//static bool Read(int32 iRank, uint32 *pBitfield)
//{
//    // broadcast send index if target node
//    int32 iNodeIndex = kCellNotFound;
//    MPI_Bcast(&iNodeIndex, 1, MPI_INT, s_iMpiHostRank, MPI_COMM_WORLD);
//
//    if (iNodeIndex == kCellNotFound)
//        return false;
//    
//    // host node can store data itself
//    if (iRank == iNodeIndex)
//    {
//        assert(s_iDeviceIndex != kCellNotFound);
//        assert(s_iCellIndex != kCellNotFound);
//
//        // activate selected device (did store in the last exec() call)
//        cudaError_t eErrorCode = cudaSetDevice(s_iDeviceIndex);
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        // get data directly from device
//        const DeviceConfiguration &roDeviceConfiguration = s_oNodeConfiguration[s_iDeviceIndex];
//        eErrorCode = cudaMemcpy(
//            pBitfield,
//            d_aMemory[s_iDeviceIndex] + s_iCellIndex * roDeviceConfiguration.m_uiCellDim,
//            roDeviceConfiguration.m_uiCellDim * sizeof(uint32),
//            cudaMemcpyDeviceToHost
//        );
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        // send data if needed
//        if (iRank != s_iMpiHostRank)
//            MPI_Send(pBitfield, roDeviceConfiguration.m_uiCellDim * sizeof(uint32), MPI_BYTE, s_iMpiHostRank, MPI_ANY_TAG, MPI_COMM_WORLD);
//    }
//
//    if (iRank == s_iMpiHostRank && iRank != iNodeIndex)
//    {
//        int32 iSize = s_oNodeConfiguration[0].m_uiCellDim * sizeof(uint32);
//        MPI_Recv(pBitfield, iSize, MPI_BYTE, iNodeIndex, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    }
//
//    return true;
//}