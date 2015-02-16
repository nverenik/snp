#include "Worker.h"

#include "../network/ProtocolHandler.h"
#include "../network/Packet.h"
#include "../network/RenameMe.h"

#include <map>
#include <math.h>

#include <cuda_runtime.h>
#include "Kernel.h"

NS_SNP_BEGIN

bool CWorker::IsCUDASupported()
{
    return false;
    //int iDeviceCount = 0;
    //cudaError_t eErrorCode = cudaGetDeviceCount(&iDeviceCount);
    //if (eErrorCode != cudaSuccess)
    //    MPI_LOG("CUDA error: %s", cudaGetErrorString(eErrorCode));
    //else
    //    MPI_LOG("Number of devices: %d", iDeviceCount);
}

CWorker::CWorker(MPI_Comm oCommunicator, int32 iHostRank)
    : m_oCommunicator(oCommunicator)
    , m_iHostRank(iHostRank)
    , m_bShouldExit(false)
    , m_bRunning(false)
    , m_uiCellSize(0)
    , m_iNodeIndex(kCellNotFound)
    , m_iDeviceIndex(kCellNotFound)
    , m_iCellIndex(kCellNotFound)
{
    MPI_Comm_size(m_oCommunicator, &m_iGroupSize);
    MPI_Comm_rank(m_oCommunicator, &m_iRank);
    m_bHost = (m_iRank == m_iHostRank);

    int32 iLength = 0;
    MPI_Get_processor_name(m_pszProcessorName, &iLength);
}

bool CWorker::Init()
{
    // at first each mpi node collect information about available GPU devices
    int32 iDeviceCount = 0;
    cudaGetDeviceCount(&iDeviceCount);

    // total number of available devices shared between all nodes
    // prepare buffer for receiving
    std::vector<int32> aiDeviceCount;
    aiDeviceCount.resize(GetGroupSize());

    // exchange information
    MPI_Allgather(&iDeviceCount, 1, MPI_INT, &aiDeviceCount.front(), 1, MPI_INT, GetCommunicator());
    
    // there're no devices on this or some else node
    for (int32 iNodeRank = 0; iNodeRank < GetGroupSize(); iNodeRank++)
    {
        // it's error, must be avoided in the beginning while configuring communicator group
        if (!aiDeviceCount[iNodeRank])
            return false;
    }

    // store information about available devices
    m_oNodeInfo.resize(iDeviceCount);
    for (int32 iDeviceIndex = 0; iDeviceIndex < iDeviceCount; iDeviceIndex++)
        cudaGetDeviceProperties(&m_oNodeInfo[iDeviceIndex], iDeviceIndex);

    // threat device info struct as raw array of bytes
    MPI_Datatype iDeviceInfoType;
    MPI_Type_contiguous(sizeof(tDeviceInfo), MPI_BYTE, &iDeviceInfoType);
    MPI_Type_commit(&iDeviceInfoType);

    if (IsHost())
    {
        // prepare receiving buffer to store information about the whole system
        m_oSystemInfo.clear();
        m_oSystemInfo.resize(GetGroupSize());

        for (int32 iNodeRank = 0; iNodeRank < GetGroupSize(); iNodeRank++)
            m_oSystemInfo[iNodeRank].resize(aiDeviceCount[iNodeRank]);

        // fill the info about host in advance (no need to send it)
        m_oSystemInfo[GetHostRank()] = m_oNodeInfo;
    }
    
    if (!IsHost())
    {
        // send current node information to the host
        MPI_Send(&m_oNodeInfo.front(), (int32)m_oNodeInfo.size(), iDeviceInfoType, GetHostRank(), 0, GetCommunicator());
    }

    if (IsHost() && GetGroupSize() > 1)
    {
        // receive information on the host side
        std::vector<MPI_Request> aRequests;
        for (int32 iNodeRank = 0; iNodeRank < GetGroupSize(); iNodeRank++)
        {
            if (iNodeRank == GetHostRank())
                continue;

            // using non blocking receiving
            MPI_Request iRequest = 0;
            MPI_Irecv(&m_oSystemInfo[iNodeRank].front(), (int32)m_oSystemInfo[iNodeRank].size(), iDeviceInfoType, iNodeRank, 0, GetCommunicator(), &iRequest);
            aRequests.push_back(iRequest);
        }
        MPI_Waitall((int32)aRequests.size(), &aRequests.front(), MPI_STATUS_IGNORE);
    }

    MPI_Type_free(&iDeviceInfoType);
    return true;
}

bool CWorker::PrintSystemInfo() const
{
    if (!IsHost())
        return false;

    for (int32 iNodeRank = 0; iNodeRank < m_oSystemInfo.size(); iNodeRank++)
    {
        printf("\n");
        printf("=============== Node #%d ===============\n", iNodeRank);

        const tNodeInfo &roNodeInfo = m_oSystemInfo[iNodeRank];
        for (int32 iDeviceIndex = 0; iDeviceIndex < roNodeInfo.size(); iDeviceIndex++)
        {
            const cudaDeviceProp &roDeviceProps = roNodeInfo[iDeviceIndex];

            printf("\n");
            printf("[Device #%d]", iDeviceIndex);
            printf("\n");

            // General
            printf("   Device name: %s\n", roDeviceProps.name);
            printf("   Clock rate: %d\n", roDeviceProps.clockRate);
            printf("\n");

            // Computing
            printf("   Multiprocessor count: %d\n", roDeviceProps.multiProcessorCount);
            printf("   Max Threads per Multiprocessor: %d\n", roDeviceProps.maxThreadsPerMultiProcessor);
            printf("   Max grid size (x;y;z): (%d;%d;%d)\n",
                roDeviceProps.maxGridSize[0],
                roDeviceProps.maxGridSize[1],
                roDeviceProps.maxGridSize[2]);
            printf("   Max block size (x;y;z): (%d;%d;%d)\n",
                roDeviceProps.maxThreadsDim[0],
                roDeviceProps.maxThreadsDim[1],
                roDeviceProps.maxThreadsDim[2]);
            printf("   Warp size: %d\n", roDeviceProps.warpSize);
            printf("\n");

            // Memory
            printf("   Total global memory: %zu\n", roDeviceProps.totalGlobalMem);
            printf("   Total constant memory: %zu\n", roDeviceProps.totalConstMem);
            printf("   Shared memory per block: %zu\n", roDeviceProps.sharedMemPerBlock);
            printf("   Registers per block: %d\n", roDeviceProps.regsPerBlock);
            printf("\n");
        }
    }

    // TODO: review these recomendations
    //// Recommendations
    //printf("   Min threads: %d\n", iDeviceCount * roDeviceProps.multiProcessorCount * roDeviceProps.maxThreadsPerMultiProcessor);    // Num of threads that may run at once
    //printf("   Max threads: %d\n", iDeviceCount * roDeviceProps.maxGridSize[0] * roDeviceProps.maxThreadsDim[0]);                    // Logical limit
    //printf("   Memory for Min: %d\n", roDeviceProps.totalGlobalMem / (roDeviceProps.multiProcessorCount * roDeviceProps.maxThreadsPerMultiProcessor));
    //printf("   Memory for Max: %d\n", roDeviceProps.totalGlobalMem / (roDeviceProps.maxGridSize[0] * roDeviceProps.maxThreadsDim[0]));
    return true;
}

void CWorker::RunLoop(CProtocolHandler *pHandler)
{
    typedef tPacket::tData tData;
    typedef tPacket::tDynamicData tDynamicData;
    typedef std::map<tPacket::tType, CWorker::tCommand> tCommandMap;

    tCommandMap oCommandMap;
    oCommandMap[tPacket::tType_RequestStartup]  = CWorker::tCommand_Startup;
    oCommandMap[tPacket::tType_RequestExec]     = CWorker::tCommand_Exec;
    oCommandMap[tPacket::tType_RequestRead]     = CWorker::tCommand_Read;
    oCommandMap[tPacket::tType_RequestShutdown] = CWorker::tCommand_Shutdown;

    MPI_Datatype iDataType;
    MPI_Type_contiguous(sizeof(tData), MPI_BYTE, &iDataType);
    MPI_Type_commit(&iDataType);

    while(!m_bShouldExit)
    {
        // here the current command with parameters
        tCommand eCommand = tCommand_Idle;
        tData oData;
        tDynamicData oDynamicData;

        // host should initialize them...
        if (IsHost())
        {
            //...using data received from the main app
            tPacket oPacket = pHandler->Read(); // blocking method
            assert(oCommandMap.count(oPacket.m_eType) > 0);

            eCommand = oCommandMap[oPacket.m_eType];
            oData = oPacket.m_oData;
            oDynamicData.swap(oPacket.m_oDynamicData);
        }

        // broadcast command to all mpi nodes
        MPI_Bcast(&eCommand, 1, MPI_INT, GetHostRank(), GetCommunicator());
        assert(eCommand != tCommand_Idle);

        // broadcast command parameters
        MPI_Bcast(&oData, 1, iDataType, GetHostRank(), GetCommunicator());

        oDynamicData.push_back(0);  // to be sure that vector is not empty
        MPI_Bcast(&oDynamicData.front(), oDynamicData.size(), MPI_BYTE, GetHostRank(), GetCommunicator());
        oDynamicData.erase(oDynamicData.begin());

        LOG_MESSAGE(5, "Processing command %d", eCommand);
        switch(eCommand)
        {
            case tCommand_Startup:
            {
                // these parameters can be changed in startup so we
                // will send the new values to the main app
                uint16 uiCellSize = oData.asRequestStartup.m_uiCellSize;
                uint32 uiCellsPerPU = oData.asRequestStartup.m_uiCellsPerPU;
                uint32 uiNumberOfPU = oData.asRequestStartup.m_uiNumberOfPU;

                assert(!m_bRunning);
                bool bResult = Startup(uiCellSize, uiCellsPerPU, uiNumberOfPU);
                if (IsHost())
                {
                    tPacket oPacket;
                    oPacket.m_eType = tPacket::tType_ResponseStartup;
                    oPacket.m_oData.asResponseStartup.m_bResult = bResult;
                    oPacket.m_oData.asResponseStartup.m_uiCellSize = uiCellSize;
                    oPacket.m_oData.asResponseStartup.m_uiCellsPerPU = uiCellsPerPU;
                    oPacket.m_oData.asResponseStartup.m_uiNumberOfPU = uiNumberOfPU;
                    pHandler->Write(&oPacket);
                }

                if (bResult)
                {
                    m_uiCellSize = uiCellSize;
                    m_bRunning = bResult;
                }
                break;
            };
            
            case tCommand_Exec:
            {
                assert(m_bRunning && oDynamicData.size() > 0);
                bool bResult = Exec(
                    oData.asRequestExec.m_bSingleCell,
                    oData.asRequestExec.m_eOperation,
                    (uint32 *)&oDynamicData.front()
                );
                if (IsHost())
                {
                    tPacket oPacket;
                    oPacket.m_eType = tPacket::tType_ResponseExec;
                    oPacket.m_oData.asResponseExec.m_bResult = bResult;
                    pHandler->Write(&oPacket);
                }
                break;
            };

            case tCommand_Read:
            {
                assert(m_bRunning && m_uiCellSize > 0);
                std::vector<uint32> aOutput;
                aOutput.resize(m_uiCellSize);

                bool bResult = Read(&aOutput.front());
                if (IsHost())
                {
                    tPacket oPacket;
                    oPacket.m_eType = tPacket::tType_ResponseRead;
                    oPacket.m_oData.asResponseRead.m_bResult = bResult;
                    if (bResult)
                    {
                        oPacket.m_oDynamicData.resize(m_uiCellSize * sizeof(uint32));
                        memcpy(&oPacket.m_oDynamicData.front(), &aOutput.front(), m_uiCellSize * sizeof(uint32));
                    }
                    pHandler->Write(&oPacket);
                }
                break;
            };

            case tCommand_Shutdown:
            {
                assert(m_bRunning);
                bool bResult = Shutdown();
                if (IsHost())
                {
                    tPacket oPacket;
                    oPacket.m_eType = tPacket::tType_ResponseShutdown;
                    oPacket.m_oData.asResponseShutdown.m_bResult = bResult;
                    pHandler->Write(&oPacket);
                }

                // break main loop
                m_bRunning = false;
                m_bShouldExit = true;

                LOG_MESSAGE(1, "System shutdown.");
                break;
            };

            default: break;
        }
    }

    MPI_Type_free(&iDataType);
}

bool CWorker::Startup(uint16 &uiCellSize, uint32 &uiCellsPerPU, uint32 &uiNumberOfPU)
{
    assert(!d_aMemory.size());
    assert(!d_aInstruction.size());
    assert(!d_aOutput.size());

    assert(!h_aOutput.size());
    assert(!h_aCell.size());

    // todo: we can change uiCellSize and uiCellsPerPU depending on hardware

    // Find the configuration for GPUs
    tSystemConfig oSystemConfig;
    oSystemConfig.resize(m_oSystemInfo.size());

    // (!)Assume that each device (GPU) is used maximum only by only one node
    if (IsHost())
    {
        // 1. number of blocks is multiple of multiprocessors amount
        // 2. as number of threads per block use the maximum
        uint32 uiNumberOfThreadsPerIteration = 0;
        for (uint32 iNodeIndex = 0; iNodeIndex < m_oSystemInfo.size(); iNodeIndex++)
        {
            const tNodeInfo &roNodeInfo = m_oSystemInfo[iNodeIndex];
            for (uint32 iDeviceIndex = 0; iDeviceIndex < roNodeInfo.size(); iDeviceIndex++)
            {
                // for each iteration add number of threads equals to what we obtain if add
                // 1 block for each multi processor and use maximum threads in this block
                uiNumberOfThreadsPerIteration += 
                    roNodeInfo[iDeviceIndex].multiProcessorCount *
                    roNodeInfo[iDeviceIndex].maxThreadsDim[0];
            }
        }

        // find the minimum configuration which covers requested memory volume
        uint32 uiMultiplier = uint32(ceilf((float)uiNumberOfPU / uiNumberOfThreadsPerIteration));
        for (uint32 iNodeIndex = 0; iNodeIndex < oSystemConfig.size(); iNodeIndex++)
        {
            tNodeInfo &roNodeInfo = m_oSystemInfo[iNodeIndex];
            tNodeConfig &roNodeConfig = oSystemConfig[iNodeIndex];
            roNodeConfig.resize(roNodeInfo.size());

            for (uint32 iDeviceIndex = 0; iDeviceIndex < roNodeConfig.size(); iDeviceIndex++)
            {
                tDeviceConfig &roDeviceConfig = roNodeConfig[iDeviceIndex];
                roDeviceConfig.m_uiCellDim = uiCellSize;
                roDeviceConfig.m_uiThreadDim = uiCellsPerPU;
                roDeviceConfig.m_uiBlockDim = roNodeInfo[iDeviceIndex].maxThreadsDim[0];
                roDeviceConfig.m_uiGridDim = roNodeInfo[iDeviceIndex].multiProcessorCount * uiMultiplier;
            }
        }
    }

    // send configurations to each process, prepare receiving buffer
    const uint32 uiNumberOfLocalDevices = uint32(m_oNodeInfo.size());
    m_oNodeConfig.clear();
    m_oNodeConfig.resize(uiNumberOfLocalDevices);

    // threat device info struct as raw array of bytes
    MPI_Datatype iDeviceConfigType;
    MPI_Type_contiguous(sizeof(tDeviceConfig), MPI_BYTE, &iDeviceConfigType);
    MPI_Type_commit(&iDeviceConfigType);

    if (!IsHost())
    {
        // receive current node config from the host
        MPI_Recv(&m_oNodeConfig.front(), (int32)m_oNodeConfig.size(), iDeviceConfigType, GetHostRank(), 0, GetCommunicator(), MPI_STATUS_IGNORE);
    }

    if (IsHost())
    {
        // don't send host data to itself
        m_oNodeConfig = oSystemConfig[GetHostRank()];
        // send data for each node separately
        if (GetGroupSize() > 1)
        {
            std::vector<MPI_Request> aRequests;
            for (int32 iNodeRank = 0; iNodeRank < GetGroupSize(); iNodeRank++)
            {
                if (iNodeRank == GetHostRank())
                    continue;
                
                // using non blocking method
                MPI_Request iRequest = 0;
                MPI_Isend(&oSystemConfig[iNodeRank].front(), (int32)oSystemConfig[iNodeRank].size(), iDeviceConfigType, iNodeRank, 0, GetCommunicator(), &iRequest);
                aRequests.push_back(iRequest);
            }
            MPI_Waitall((int32)aRequests.size(), &aRequests.front(), MPI_STATUS_IGNORE);
        }
    }

    MPI_Type_free(&iDeviceConfigType);

    // allocate processor memory separately for each GPU
    d_aMemory.resize(uiNumberOfLocalDevices);
    d_aInstruction.resize(uiNumberOfLocalDevices);
    d_aOutput.resize(uiNumberOfLocalDevices);

    h_aOutput.resize(uiNumberOfLocalDevices);
    h_aCell.resize(uiNumberOfLocalDevices);

    cudaError_t eErrorCode = cudaSuccess;
    for (uint32 iDeviceIndex = 0; iDeviceIndex < uiNumberOfLocalDevices; iDeviceIndex++)
    {
        eErrorCode = cudaSetDevice(iDeviceIndex);
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        const tDeviceConfig &roDeviceConfig = m_oNodeConfig[iDeviceIndex];
        const uint32 uiMemorySize =
            roDeviceConfig.m_uiCellDim *
            roDeviceConfig.m_uiThreadDim *
            roDeviceConfig.m_uiBlockDim *
            roDeviceConfig.m_uiGridDim;

        LOG_MESSAGE(3, "Configure device #%d", iDeviceIndex);
        LOG_MESSAGE(3, "   Cell dim = %u", roDeviceConfig.m_uiCellDim);
        LOG_MESSAGE(3, "   Thread dim = %u", roDeviceConfig.m_uiThreadDim);
        LOG_MESSAGE(3, "   Block dim = %u", roDeviceConfig.m_uiBlockDim);
        LOG_MESSAGE(3, "   Grid dim = %u", roDeviceConfig.m_uiGridDim);
        LOG_MESSAGE(3, "Memory allocated = %lu", uiMemorySize * sizeof(uint32));

        eErrorCode = cudaMalloc((void**)&d_aMemory[iDeviceIndex], uiMemorySize * sizeof(uint32));
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        // TODO: should we place instruction and output arrays into shared memory or something for speadup?
        eErrorCode = cudaMalloc((void**)&d_aInstruction[iDeviceIndex], 4 * roDeviceConfig.m_uiCellDim * sizeof(uint32));
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        eErrorCode = cudaMalloc((void**)&d_aOutput[iDeviceIndex], roDeviceConfig.m_uiBlockDim * roDeviceConfig.m_uiGridDim * sizeof(int32));
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        // allocate buffer memory for output array
        h_aOutput[iDeviceIndex] = new int32[roDeviceConfig.m_uiBlockDim * roDeviceConfig.m_uiGridDim];
        h_aCell[iDeviceIndex] = new uint32[roDeviceConfig.m_uiCellDim];
    }

    if (IsHost())
    {
        // calculate total number of PU
        uiNumberOfPU = 0;
        for (uint32 iNodeIndex = 0; iNodeIndex < oSystemConfig.size(); iNodeIndex++)
        {
            const tNodeConfig &roNodeConfig = oSystemConfig[iNodeIndex];
            for (uint32 iDeviceIndex = 0; iDeviceIndex < roNodeConfig.size(); iDeviceIndex++)
            {
                uiNumberOfPU +=
                    roNodeConfig[iDeviceIndex].m_uiBlockDim *
                    roNodeConfig[iDeviceIndex].m_uiGridDim;
            }
        }
    }

    MPI_Barrier(GetCommunicator());
    return true;
}

bool CWorker::Shutdown()
{
    assert(d_aMemory.size());
    assert(d_aInstruction.size());
    assert(d_aOutput.size());

    assert(h_aOutput.size());
    assert(h_aCell.size());

    cudaError_t eErrorCode = cudaSuccess;
    for (uint32 iDeviceIndex = 0; iDeviceIndex < m_oNodeConfig.size(); iDeviceIndex++)
    {
        eErrorCode = cudaSetDevice(iDeviceIndex);
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        eErrorCode = cudaFree(d_aMemory[iDeviceIndex]);
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        eErrorCode = cudaFree(d_aInstruction[iDeviceIndex]);
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        eErrorCode = cudaFree(d_aOutput[iDeviceIndex]);
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        eErrorCode = cudaDeviceReset();
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        delete h_aOutput[iDeviceIndex];
        delete h_aCell[iDeviceIndex];
    }

    d_aMemory.clear();
    d_aInstruction.clear();
    d_aOutput.clear();

    h_aOutput.clear();
    h_aCell.clear();

    MPI_Barrier(GetCommunicator());
    return true;
}

bool CWorker::Exec(bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction)
{
    m_iNodeIndex = kCellNotFound;
    m_iDeviceIndex = kCellNotFound;
    m_iCellIndex = kCellNotFound;

    return (bSingleCell)
        ? ExecSequential(eOperation, pInstruction)
        : ExecParallel(eOperation, pInstruction);
}

bool CWorker::Read(uint32 *pBitfield)
{
    // broadcast rank of target mpi node (the last activated one)
    uint32 iNodeRank = m_iNodeIndex; // is valid only on host node
    MPI_Bcast(&iNodeRank, 1, MPI_INT, GetHostRank(), GetCommunicator());

    // there's no cell was activated during the last instruction
    if (iNodeRank == kCellNotFound)
        return false;

    // interrupt all node except of host and target one
    // (note that host can be target as well)
    if (!IsHost() && iNodeRank != GetRank())
        return false;

    if (iNodeRank == GetRank())
    {
        assert(m_iDeviceIndex != kCellNotFound);
        assert(m_iCellIndex != kCellNotFound);

        // activate selected device
        cudaError_t eErrorCode = cudaSetDevice(m_iDeviceIndex);
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        // get data directly from this device
        eErrorCode = cudaMemcpy(
            pBitfield,
            d_aMemory[m_iDeviceIndex] + m_iCellIndex * m_uiCellSize,
            m_uiCellSize * sizeof(uint32),
            cudaMemcpyDeviceToHost
        );

        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        // send data if needed
        if (!IsHost())
            MPI_Send(pBitfield, m_uiCellSize * sizeof(uint32), MPI_BYTE, GetHostRank(), 0, GetCommunicator());
    }

    // receive data if needed
    if (IsHost() && iNodeRank != GetRank())
        MPI_Recv(pBitfield, m_uiCellSize * sizeof(uint32), MPI_BYTE, iNodeRank, 0, GetCommunicator(), MPI_STATUS_IGNORE);
    
    return true;
}

bool CWorker::ExecSequential(tOperation eOperation, const uint32 * const pInstruction)
{
    assert(m_iNodeIndex == kCellNotFound);

    // All mpi nodes must execute instruction one by one, as only one cell can be activated
    // on the most prioritized node
    //bool bFound = false;

    // find the first mpi node with which contains matched cell
    for (int32 iNodeRank = 0; iNodeRank < GetGroupSize(); iNodeRank++)
    {
        bool bNeedsExec = false;
        if (IsHost())
        {
            // host sends command to start kernel for each mpi node
            if (iNodeRank == GetHostRank())
            {
                // no need send data to itself
                bNeedsExec = true;
            }
            else
            {
                bool bTrue = true;
                MPI_Send(&bTrue, 1, MPI_BYTE, iNodeRank, 0, GetCommunicator());
            }
        }

        // result for the current mpi node
        bool bFoundInNode = false;

        // only once per loop logic
        if (iNodeRank == GetRank())
        {
            if (!IsHost())
                MPI_Recv(&bNeedsExec, 1, MPI_BYTE, GetHostRank(), 0, GetCommunicator(), MPI_STATUS_IGNORE);

            bool bFound = false;
            if (bNeedsExec)
            {
                // so execute instruction on device
                bFound = ExecImpl(true, eOperation, pInstruction);
                if (!IsHost())
                {
                    // and send result to the host
                    MPI_Send(&bFound, 1, MPI_BYTE, GetHostRank(), 0, GetCommunicator());
                }
                else
                {
                    // again, in case of host no need to send data to itself
                    bFoundInNode = bFound;
                }
            }

            // no need to finish loop in child hodes
            if (!IsHost()) return bFound;
        }

        if (IsHost())
        {
            if (iNodeRank != GetHostRank())
            {
                // receive result from the current mpi node
                MPI_Recv(&bFoundInNode, 1, MPI_BYTE, iNodeRank, 0, GetCommunicator(), MPI_STATUS_IGNORE);
            }

            // current node contains matched cell, so no need to execute kernel further
            if (bFoundInNode)
            {
                m_iNodeIndex = iNodeRank;
                // but we need send exec command for all remaining mpi nodes
                iNodeRank += 1;
                for (; iNodeRank < GetGroupSize(); iNodeRank++)
                {
                    bool bFalse = false;
                    MPI_Send(&bFalse, 1, MPI_BYTE, iNodeRank, 0, GetCommunicator());
                }
                break;
            }
        }
    }

    assert(IsHost());
    return (m_iNodeIndex != kCellNotFound);
}

bool CWorker::ExecParallel(tOperation eOperation, const uint32 * const pInstruction)
{
    assert(m_iNodeIndex == kCellNotFound);

    // note: somehow MPI_Gather doesn't support bool type
    struct tBool {
        bool m_bValue;
    };

    // All mpi nodes can execute instruction in parallel
    tBool oFound;
    oFound.m_bValue = ExecImpl(false, eOperation, pInstruction);

    // host collects result data, prepare receiving buffer

    std::vector<tBool> aFound;
    if (IsHost())
        aFound.resize(GetGroupSize());
    
    MPI_Datatype iBoolType;
    MPI_Type_contiguous(sizeof(tBool), MPI_BYTE, &iBoolType);
    MPI_Type_commit(&iBoolType);

    MPI_Gather(&oFound, 1, iBoolType, &aFound.front(), 1, iBoolType, GetHostRank(), GetCommunicator());
    MPI_Type_free(&iBoolType);

    if (IsHost())
    {
        // find the first mpi node with which contains matched cell
        m_iNodeIndex = kCellNotFound;
        for (uint32 iNodeIndex = 0; iNodeIndex < aFound.size(); iNodeIndex++)
        {
            if (aFound[iNodeIndex].m_bValue)
            {
                m_iNodeIndex = iNodeIndex;
                break;
            }
        }
        return (m_iNodeIndex != kCellNotFound);
    }
    return oFound.m_bValue;
}

bool CWorker::ExecImpl(bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction)
{
    assert(m_iDeviceIndex == kCellNotFound);
    assert(m_iCellIndex == kCellNotFound);

    // execute kernel function for each device
    for (uint32 iDeviceIndex = 0; iDeviceIndex < m_oNodeConfig.size(); iDeviceIndex++)
    {
        cudaError_t eErrorCode = cudaSetDevice(iDeviceIndex);
        if (eErrorCode != cudaSuccess)
            LOG_MESSAGE(1, "CUDA error: %s", cudaGetErrorString(eErrorCode));

        // todo: how to execute kernel for each device in parallel? is separated thread needed?
        const tDeviceConfig &roDeviceConfig = m_oNodeConfig[iDeviceIndex];
        uint32 iCellIndex = kernel_exec(
            bSingleCell,
            eOperation,
            pInstruction,
            roDeviceConfig.m_uiCellDim,
            roDeviceConfig.m_uiThreadDim,
            roDeviceConfig.m_uiBlockDim,
            roDeviceConfig.m_uiGridDim,
            d_aMemory[iDeviceIndex],
            d_aInstruction[iDeviceIndex],
            d_aOutput[iDeviceIndex],
            h_aOutput[iDeviceIndex],
            h_aCell[iDeviceIndex]
        );

        if (m_iCellIndex == kCellNotFound)
        {
            // save the first matched cell
            m_iDeviceIndex = iDeviceIndex;
            m_iCellIndex = iCellIndex;

            // if instruction is performed only on the first matched cell
            // we can not run kernel on the next devices
            if (bSingleCell) break;
        }
    }

    return (m_iCellIndex != kCellNotFound);
}

NS_SNP_END
