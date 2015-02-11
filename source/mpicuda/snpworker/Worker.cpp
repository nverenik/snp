#include "Worker.h"
#include <cuda_runtime.h>

//#include "../ProtocolHandler.h"
#include "../Packet.h"

#include <map>

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
    MPI_Type_contiguous(sizeof(snpDeviceInfo), MPI_BYTE, &iDeviceInfoType);
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
        MPI_Send(&m_oNodeInfo.front(), m_oNodeInfo.size(), iDeviceInfoType, GetHostRank(), 0, GetCommunicator());
    }

    if (IsHost() && GetGroupSize() > 1)
    {
        // receive information on the host side
        std::vector<MPI_Request> apRequests;
        for (int32 iNodeRank = 0; iNodeRank < GetGroupSize(); iNodeRank++)
        {
            if (iNodeRank == GetHostRank())
                continue;

            // using non blocking receiving
            MPI_Request pRequest = 0;
            MPI_Irecv(&m_oSystemInfo[iNodeRank].front(), m_oSystemInfo[iNodeRank].size(), iDeviceInfoType, iNodeRank, 0, GetCommunicator(), &pRequest);
            apRequests.push_back(pRequest);
        }
        MPI_Waitall(apRequests.size(), &apRequests.front(), MPI_STATUS_IGNORE);
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

        const snpNodeInfo &roNodeInfo = m_oSystemInfo[iNodeRank];
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
}

void CWorker::RunLoop()
{
    typedef tPacket::tData tData;

    class CWorkerProtocolHandler// : public CProtocolHandler
    {
    public:
        CWorkerProtocolHandler()
            : m_pPacket(NULL)
        {
            m_oCommandMap[tPacket::tType_Startup]   = CWorker::tCommand_Startup;
            m_oCommandMap[tPacket::tType_Exec]      = CWorker::tCommand_Exec;
            m_oCommandMap[tPacket::tType_Read]      = CWorker::tCommand_Read;
            m_oCommandMap[tPacket::tType_Shutdown]  = CWorker::tCommand_Shutdown;
        }

        virtual ~CWorkerProtocolHandler()
        {
            if (m_pPacket)
                delete m_pPacket;
        }

        inline tCommand ReadCommand() const
        {
            if (!m_pPacket || !m_oCommandMap.count(m_pPacket->m_eType))
                return CWorker::tCommand_Idle;
            return m_oCommandMap.at(m_pPacket->m_eType);
        }

        inline tData * ReadData() const
        {
            return (m_pPacket) ? &m_pPacket->m_oData : NULL;
        }

        inline void NextCommand()
        {
            if (m_pPacket)
            {
                delete m_pPacket;
                m_pPacket = NULL;
            }
        }

        // todo: parent methods - remove placeholders later
        void Tick() { Execute(); }
        tPacket * GrabPacket() { return NULL; }

        inline void Execute()
        {
            if (!m_pPacket)
                m_pPacket = GrabPacket();
        }

    private:
        typedef std::map<tPacket::tType, CWorker::tCommand> tCommandMap;

        tCommandMap    m_oCommandMap;
        tPacket        *m_pPacket;
    };

    // only host works with the socket
    CWorkerProtocolHandler *pHandler = IsHost() ? new CWorkerProtocolHandler() : NULL;
    while(true)
    {
        // here the current command with parameters
        tCommand eCommand = tCommand_Idle;
        tData *pData = NULL;

        // host should initialize them
        if (IsHost())
        {
            // using data received from the main app
            while(tCommand_Idle == (eCommand = pHandler->ReadCommand()))
            {
                msleep(0);
                pHandler->Tick();
            }
            pData = pHandler->ReadData();
        }

        assert(eCommand != tCommand_Idle && pData);
        //LOG_MESSAGE( 3, "Processing packet \"%s\"", pPacket->ToString().c_str() );

        // broadcast command to all mpi nodes
        MPI_Bcast(&eCommand, 1, MPI_INT, GetHostRank(), GetCommunicator());

        switch(eCommand)
        {
            case tCommand_Startup:
            {
                Startup();
                break;
            };
            
            case tCommand_Exec:
            {
                Exec();
                break;
            };

            case tCommand_Read:
            {
                Read();
                break;
            };

            case tCommand_Shutdown:
            {
                Shutdown();
                break;
            };

            default: break;
        }

        // release current packet before the next one
        pHandler->NextCommand();
    }

    if (pHandler)
    {
        delete pHandler;
        pHandler = NULL;
    }    
}

bool CWorker::Startup()
{
    return false;
}

bool CWorker::Shutdown()
{
    return false;
}

bool CWorker::Exec()
{
    return false;
}

bool CWorker::Read()
{
    return false;
}

NS_SNP_END
