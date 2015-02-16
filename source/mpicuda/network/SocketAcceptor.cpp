#include <sys/types.h>
#include <fcntl.h>

#include "SocketAcceptor.h"
#include "RenameMe.h"

CSocketAcceptor::CSocketAcceptor()
{
    m_iListeningPort = 60666;
    m_iAcceptedSocket = -1;
}

void CSocketAcceptor::AcceptConnections()
{
    int iListeningSocketFD, iClientSocketFD;
    struct sockaddr_in oSrvAddr, oClientAddr;

#ifdef WIN32
    int iInvalidSocket = INVALID_SOCKET;
#else
    int iInvalidSocket = -1;
#endif

    if ((iListeningSocketFD = socket(AF_INET, SOCK_STREAM, 0)) == iInvalidSocket)
    {
        LOG_MESSAGE(1, "Socket creation error: %s", strerror(errno));
        return;
    }

    memset( &oSrvAddr, 0, sizeof(oSrvAddr) );
    oSrvAddr.sin_family = AF_INET;
    oSrvAddr.sin_addr.s_addr = INADDR_ANY;
    oSrvAddr.sin_port = htons(m_iListeningPort);
    int iSockOption = 1;
    if(SetSocketOptions(iListeningSocketFD, SOL_SOCKET, SO_REUSEADDR, &iSockOption, sizeof(int)) == -1)
    {
        LOG_MESSAGE( 1, "setsockopt error for socket[FD=%d], port[%d]: %s", iListeningSocketFD, m_iListeningPort, strerror(errno) );
        CloseSocket(iListeningSocketFD);
        return;
    }

    if(bind(iListeningSocketFD, (struct sockaddr*) &oSrvAddr, sizeof(oSrvAddr)) == -1)
    {
        LOG_MESSAGE( 1, "bind error for socket[FD=%d]: %s", iListeningSocketFD, strerror(errno) );
        CloseSocket(iListeningSocketFD);
        return;
    }

    if(listen(iListeningSocketFD, 1024) == -1)
    {
        LOG_MESSAGE( 1, "listen error for socket[FD=%d]: %s", iListeningSocketFD, strerror(errno) );
        CloseSocket(iListeningSocketFD);
        return;
    }
    else
    {
        LOG_MESSAGE(3, "Listening port[%d]", m_iListeningPort);
    }

    while(true)
    {
        socklen_t iClientAddrLen = sizeof(oClientAddr);
        if( (iClientSocketFD = accept(iListeningSocketFD, (struct sockaddr*) &oClientAddr, &iClientAddrLen)) == -1 )
        {
            LOG_MESSAGE( 1, "accept error for socket[FD=%d]: %s", iListeningSocketFD, strerror(errno) );
        }
        else
        {
            m_iAcceptedSocket = iClientSocketFD;
            break;
        }
    }

    if(iListeningSocketFD)
        CloseSocket(iListeningSocketFD);

    LOG_MESSAGE(2, "Acceptor stopped");
}
