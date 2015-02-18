#include <mpi.h>

#include <stdio.h>
#include <memory>

#include <tclap/CmdLine.h>

#ifdef WIN32
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#endif // WIN32

#include "../network/DataTypes.h"
#include "../network/RenameMe.h"
#include "../network/ProtocolHandler.h"

#include "Worker.h"

struct tConfig;

static void OnExit();
static bool ProcessCommandLine(int32 argc, char* argv[], tConfig &roConfig);
static int32 Connect(const std::string &sHost, int32 iPort);

extern "C" void * ThreadServerF(void *pArg);

struct tConfig
{
    std::string m_sHostDefault;
    std::string m_sHost;
    int32       m_iPortDefault;
    int32       m_iPort;
    bool        m_bTestEnabled;
    bool        m_bLogSystemInfo;
};

int main(int argc, char* argv[])
{
    ::atexit(OnExit);
    MPI_Init(&argc, &argv);

    // TODO: replace communicator by group of nodes with suitable GPUs
    snp::CWorker oWorker(/*communicator handler*/MPI_COMM_WORLD, /*host process rank*/0);

    bool bExit = false;

    // host node parses command line
    tConfig oConfig;
    oConfig.m_sHostDefault = "127.0.0.1";
    oConfig.m_iPortDefault = 60666;

    if (oWorker.IsHost() && !ProcessCommandLine(argc, argv, oConfig))
        bExit = true;

    // and finish all processes in case of error
    MPI_Bcast(&bExit, 1, MPI_BYTE, oWorker.GetHostRank(), oWorker.GetCommunicator());
    if (bExit) return 0;

    oWorker.Init();
    if (oWorker.IsHost() && oConfig.m_bLogSystemInfo)
    {
        oWorker.PrintSystemInfo();
        bExit = true;
    }       

    MPI_Bcast(&bExit, 1, MPI_BYTE, oWorker.GetHostRank(), oWorker.GetCommunicator());
    if (bExit) return 0;

    system("hostname");

    // if test mode is enabled host run the separated thread which emulates server-side
    pthread_t hThreadServer;
    if (oWorker.IsHost() && oConfig.m_bTestEnabled)
    {
        // run separated thread which emulates main app
        if (pthread_create(&hThreadServer, nullptr, ThreadServerF, nullptr) != 0)
        {
            LOG_MESSAGE(1, "Error creating test server thread: %s", strerror(errno));
            bExit = true;
        }
    }

    MPI_Bcast(&bExit, 1, MPI_BYTE, oWorker.GetHostRank(), oWorker.GetCommunicator());
    if (bExit) return 0;

    // host connects to the main application
    int32 iSocketFD = -1;
    if (oWorker.IsHost())
    {
        const uint32 uiTimeout = 10 * 1000; // 10 sec
        const uint32 uiTick = 1000; // 1 sec

        uint32 uiTimer = 0;
        while(true)
        {
            iSocketFD = Connect(oConfig.m_sHost.c_str(), oConfig.m_iPort);
            if (iSocketFD != -1)
                break;

            LOG_MESSAGE(1, "Connection to the server failed: %s", strerror(errno));

            msleep(uiTick);
            uiTimer += uiTick;

            if (uiTimer >= uiTimeout)
            {
                LOG_MESSAGE(1, "Connection timeout.");
                bExit = true;
                break;
            }
        }
    }

    MPI_Bcast(&bExit, 1, MPI_BYTE, oWorker.GetHostRank(), oWorker.GetCommunicator());
    if (bExit) return 0;

    if (oWorker.IsHost())
    {
        CProtocolHandler oHanler(iSocketFD);
        oWorker.RunLoop(&oHanler);
    }
    else
    {
        oWorker.RunLoop(nullptr);
    }

    if (oWorker.IsHost() && oConfig.m_bTestEnabled)
    {
        // wait until child thread is finished
        if ((pthread_join(hThreadServer, NULL)) != 0)
            LOG_MESSAGE(1, "pthread_join error: %s", strerror(errno));
    }

    // MPI_Finalize() is called using atexit callback
    return 0;
}

static void OnExit()
{
    MPI_Finalize();
}

static bool ProcessCommandLine(int argc, char* argv[], tConfig &roConfig)
{
    try
    {
        TCLAP::CmdLine oCommandLine("The worker executable is a part of software imitation model of the associative SNP "
            "(Semantic Network Processor, see http://github.com/nverenik/snp). It's responsible "
            "for connection with the main application and execution received commands on the "
            "computation cluster using MPI and NVidia CUDA frameworks.", ' ', "0.1.0");

        TCLAP::SwitchArg oTestSwitch("", "test", "Runs host worker in the test mode. It will generate a sequence of dummy "
            "commands and send them to the nodes.");
        TCLAP::SwitchArg oInfoSwitch("", "info", "Displays detailed cluster information and exits.", oCommandLine, false);

        TCLAP::ValueArg<int32> oPortArg("", "port", "Port number for connection to host.", false, roConfig.m_iPortDefault,
            "port number", oCommandLine);

        TCLAP::ValueArg<std::string> oHostArg("", "host", "Host address of the server side of SNP (which is included into "
            "main app) for worker executable to connect to.", true, roConfig.m_sHostDefault, "address");

        oCommandLine.xorAdd(oHostArg, oTestSwitch);
        
        // Parse the argv array.
        oCommandLine.parse(argc, argv);
        roConfig.m_sHost = oHostArg.getValue();
        roConfig.m_iPort = oPortArg.getValue();
        roConfig.m_bTestEnabled = oTestSwitch.getValue();
        roConfig.m_bLogSystemInfo = oInfoSwitch.getValue();

        // suppress test mode if server was specified
        if (roConfig.m_sHost != roConfig.m_sHostDefault)
            roConfig.m_bTestEnabled;
    }
    catch (...)
    {
        return false;
    }
    return true;
}

static int32 Connect(const std::string &sHost, int32 iPort)
{
    LOG_MESSAGE(3, "Connecting to %s:%d", sHost.c_str(), iPort);

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
