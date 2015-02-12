#ifndef _PROTOCOL_HANDLER_H_
#define _PROTOCOL_HANDLER_H_

#include "DataTypes.h"
#include "Socket.h"
#include "Packet.h"

#define IN_PACKET_BUFFER_MAX_SIZE    5
#define OUT_PACKET_BUFFER_MAX_SIZE    5

class CProtocolHandler
{
public:
    CProtocolHandler();
    virtual ~CProtocolHandler();
    void Tick();
    void InitSocket(int iSocketFD);
    void AddOutgoingPacket(tPacket* pPacket);

protected:
    tPacket* GrabPacket();

private:
    tSocket m_oSocket;
    std::list<tPacket*> m_aInPacketBuffer;
    std::list<tPacket*> m_aOutPacketBuffer;

    fd_set m_oReadSet;
    fd_set m_oWriteSet;
    fd_set m_oErrorSet;

    void Select();
    void ReadWrite();
    void Extract();
    void Pack();
    virtual void Execute() = 0;

};

#endif // _PROTOCOL_HANDLER_H_
