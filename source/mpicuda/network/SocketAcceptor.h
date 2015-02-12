#ifndef _SOCKET_ACCEPTOR_H_
#define _SOCKET_ACCEPTOR_H_

#include "DataTypes.h"
#include "Socket.h"

class CSocketAcceptor
{
public:
    CSocketAcceptor();

    void AcceptConnections();
    int GetAcceptedSocket() const { return m_iAcceptedSocket; }

private:
    int m_iListeningPort;
    int m_iAcceptedSocket;

};

#endif // _SOCKET_ACCEPTOR_H_
