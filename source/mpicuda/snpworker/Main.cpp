#include <mpi.h>

#include <stdio.h>
#include <memory>

#include <tclap/CmdLine.h>

#ifdef WIN32
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#endif // WIN32

//#include <snp/snpOperation.h>
//using snp::snpOperation;
//
//#include "../snpCommand.h"
//using snp::snpCommand;
//
//#include "Kernel.h"
#include "Worker.h"
//
struct Config;
//struct DeviceInfo;
//struct DeviceConfiguration;
//
//typedef std::vector<DeviceInfo> SystemInfo;
//typedef std::vector<DeviceConfiguration> SystemConfiguration;

static void OnExit();
static bool ProcessCommandLine(int32 argc, char* argv[], Config &roConfig);

extern "C" void * ThreadServerF(void *pArg);

struct Config
{
    bool    m_bTestEnabled;
    bool    m_bLogSystemInfo;
};

//struct DeviceInfo
//{
//    int32            m_iNodeRank;
//    cudaDeviceProp    m_oProperties;
//};
//
//struct DeviceConfiguration
//{
//    uint32    m_uiGridDim;    // number of blocks (1 .. 65536) within grid
//    uint32    m_uiBlockDim;    // number of threads (1 .. 1024) within block
//    uint32    m_uiThreadDim;    // number of cells within thread
//    uint32    m_uiCellDim;    // number of uint32 within cell
//};
//
//static const int32    s_iMpiHostRank    = 0;
//static int32        s_iMpiRank        = -1;
//static char            s_pszProcessorName[MPI_MAX_PROCESSOR_NAME];
//
//static SystemInfo            s_oNodeInfo;            // info about all available devices in the current node
//static SystemConfiguration    s_oNodeConfiguration;    // configuration for each device in the current node
//
//// pointers to the memory allocated on device
//static std::vector<uint32 *>    d_aMemory;
//static std::vector<uint32 *>    d_aInstruction;
//static std::vector<int32 *>        d_aOutput;
//
//// pre-allocated buffers used when working with kernel
//static std::vector<int32 *>        h_aOutput;
//static std::vector<uint32 *>    h_aCell;
//
//// result of ther last performed Exec() command is the index of device and index of the cell inside its memory
//// pointing to the first matched cell during instruction
//static int32    s_iNodeIndex    = kCellNotFound;
//static int32    s_iDeviceIndex    = kCellNotFound;
//static int32    s_iCellIndex    = kCellNotFound;
//
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
            return 0;
        }
    }

    // connect here?

    oWorker.RunLoop();

    //if (s_iMpiRank == s_iMpiHostRank && oConfig.m_bLogSystemInfo)
    //{
    //    PrintSystemInfo(oSystemInfo);
    //    return 0;
    //}

    ////Startup(s_iMpiRank, oSystemInfo, 9, 32, 10000);
    ////Shutdown();
    ////RunLoop(s_iMpiRank, oSystemInfo);

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

//
//static bool SendSystemInfo(const SystemInfo &roSystemInfo)
//{
//    // TODO: send system info to the main process
//    // for now this command is not used
//    return false;
//}
//
//static bool Startup(int32 iRank, const SystemInfo &roSystemInfo, uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU)
//{
//    assert(!d_aMemory.size());
//    assert(!d_aInstruction.size());
//    assert(!d_aOutput.size());
//
//    // Find the configuration for GPUs
//    SystemConfiguration oSystemConfiguration;
//
//    // (!)Assume that each device (GPU) is used maximum only by only one node
//    if (iRank == s_iMpiHostRank)
//    {
//        // 1. number of blocks is multiple of multiprocessors amount
//        // 2. as number of threads per block use the maximum
//        uint32 uiNumberOfThreadsPerIteration = 0;
//        for (uint32 iDeviceIndex = 0; iDeviceIndex < roSystemInfo.size(); iDeviceIndex++)
//        {
//            // for each iteration add number of threads equals to what we obtain if add
//            // 1 block for each multi processor and use maximum threads in this block
//            uiNumberOfThreadsPerIteration += 
//                roSystemInfo[iDeviceIndex].m_oProperties.multiProcessorCount *
//                roSystemInfo[iDeviceIndex].m_oProperties.maxThreadsDim[0];
//        }
//
//        // find the minimum configuration which covers requested memory volume
//        uint32 uiMultiplier = uint32(ceilf((float)uiNumberOfPU / uiNumberOfThreadsPerIteration));
//        for (uint32 iDeviceIndex = 0; iDeviceIndex < roSystemInfo.size(); iDeviceIndex++)
//        {
//            oSystemConfiguration.push_back(DeviceConfiguration());
//            DeviceConfiguration *pDeviceConfiguration = &oSystemConfiguration[iDeviceIndex];
//
//            pDeviceConfiguration->m_uiCellDim = uiCellSize;
//            pDeviceConfiguration->m_uiThreadDim = uiCellsPerPU;
//            pDeviceConfiguration->m_uiBlockDim = roSystemInfo[iDeviceIndex].m_oProperties.maxThreadsDim[0];
//            pDeviceConfiguration->m_uiGridDim = roSystemInfo[iDeviceIndex].m_oProperties.multiProcessorCount * uiMultiplier;
//        }
//    }
//
//    // send configurations to each process
//    const uint32 uiNumberOfLocalDevices = uint32(s_oNodeInfo.size());
//
//    s_oNodeConfiguration.clear();
//    s_oNodeConfiguration.resize(uiNumberOfLocalDevices);
//
//    // threat device info struct as raw array of bytes
//    MPI_Datatype iDeviceConfigurationType;
//    MPI_Type_contiguous(sizeof(DeviceConfiguration), MPI_BYTE, &iDeviceConfigurationType);
//    MPI_Type_commit(&iDeviceConfigurationType);
//
//    // TODO: seems MPI_Scatter do not allow to send data with different size, replace it with MPI_Send for each node
//    // collect all info on the host
//    MPI_Scatter(&oSystemConfiguration.front(), int32(oSystemConfiguration.size()), iDeviceConfigurationType,
//        &s_oNodeConfiguration.front(), uiNumberOfLocalDevices, iDeviceConfigurationType, s_iMpiHostRank, MPI_COMM_WORLD);
//    MPI_Type_free(&iDeviceConfigurationType);
//
//    // allocate processor memory separately for each GPU
//    d_aMemory.resize(uiNumberOfLocalDevices);
//    d_aInstruction.resize(uiNumberOfLocalDevices);
//    d_aOutput.resize(uiNumberOfLocalDevices);
//
//    h_aOutput.resize(uiNumberOfLocalDevices);
//    h_aCell.resize(uiNumberOfLocalDevices);
//
//    cudaError_t eErrorCode = cudaSuccess;
//    for (uint32 iDeviceIndex = 0; iDeviceIndex < uiNumberOfLocalDevices; iDeviceIndex++)
//    {
//        eErrorCode = cudaSetDevice(iDeviceIndex);
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        const DeviceConfiguration &roDeviceConfiguration = s_oNodeConfiguration[iDeviceIndex];
//        const uint32 uiMemorySize =
//            roDeviceConfiguration.m_uiCellDim *
//            roDeviceConfiguration.m_uiThreadDim *
//            roDeviceConfiguration.m_uiBlockDim *
//            roDeviceConfiguration.m_uiGridDim;
//
//        MPI_LOG("Configure device #%d", iDeviceIndex);
//        MPI_LOG("   Cell dim = %u", roDeviceConfiguration.m_uiCellDim);
//        MPI_LOG("   Thread dim = %u", roDeviceConfiguration.m_uiThreadDim);
//        MPI_LOG("   Block dim = %u", roDeviceConfiguration.m_uiBlockDim);
//        MPI_LOG("   Grid dim = %u", roDeviceConfiguration.m_uiGridDim);
//        MPI_LOG("Memory allocated = %lu", uiMemorySize * sizeof(uint32));
//
//        eErrorCode = cudaMalloc((void**)&d_aMemory[iDeviceIndex], uiMemorySize * sizeof(uint32));
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        // TODO: should we place instruction and output arrays into shared memory or something for speadup?
//        eErrorCode = cudaMalloc((void**)&d_aInstruction[iDeviceIndex], 4 * roDeviceConfiguration.m_uiCellDim * sizeof(uint32));
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        eErrorCode = cudaMalloc((void**)&d_aOutput[iDeviceIndex], roDeviceConfiguration.m_uiBlockDim * roDeviceConfiguration.m_uiGridDim * sizeof(int32));
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        // allocate buffer memory for output array
//        h_aOutput[iDeviceIndex] = new int32[roDeviceConfiguration.m_uiBlockDim * roDeviceConfiguration.m_uiGridDim];
//        h_aCell[iDeviceIndex] = new uint32[roDeviceConfiguration.m_uiCellDim];
//    }
//
//    if (iRank == s_iMpiHostRank)
//    {
//        uint32 uiTotalNumberOfPU = 0;
//        for (uint32 iDeviceIndex = 0; iDeviceIndex < oSystemConfiguration.size(); iDeviceIndex++)
//        {
//            uiTotalNumberOfPU +=
//                oSystemConfiguration[iDeviceIndex].m_uiBlockDim * 
//                oSystemConfiguration[iDeviceIndex].m_uiGridDim;
//        }
//        MPI_LOG("Total number of PU: %u", uiTotalNumberOfPU);
//    }
//
//    return true;
//}
//
//static bool Shutdown()
//{
//    cudaError_t eErrorCode = cudaSuccess;
//    for (uint32 iDeviceIndex = 0; iDeviceIndex < s_oNodeConfiguration.size(); iDeviceIndex++)
//    {
//        eErrorCode = cudaSetDevice(iDeviceIndex);
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        eErrorCode = cudaFree(d_aMemory[iDeviceIndex]);
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        eErrorCode = cudaFree(d_aInstruction[iDeviceIndex]);
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        eErrorCode = cudaFree(d_aOutput[iDeviceIndex]);
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        eErrorCode = cudaDeviceReset();
//        if (eErrorCode != cudaSuccess)
//            MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
//
//        delete h_aOutput[iDeviceIndex];
//        delete h_aCell[iDeviceIndex];
//    }
//
//    d_aMemory.clear();
//    d_aInstruction.clear();
//    d_aOutput.clear();
//
//    h_aOutput.clear();
//    h_aCell.clear();
//
//    MPI_LOG("System shutdown.");
//    return true;
//}
//
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