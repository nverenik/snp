#ifndef __PACKET_HANDLER_H__
#define __PACKET_HANDLER_H__ 

#include "../network/ProtocolHandler.h"
#include "../network/Packet.h"

#include <map>
#include <snp/snpMacros.h>

#include "Worker.h"

NS_SNP_BEGIN

typedef tPacket::tData tData;

class CWorkerProtocolHandler : public CProtocolHandler
{
public:
    CWorkerProtocolHandler()
        : m_pPacket(nullptr)
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
        return (m_pPacket) ? &m_pPacket->m_oData : nullptr;
    }

    inline void NextCommand()
    {
        if (m_pPacket)
        {
            delete m_pPacket;
            m_pPacket = nullptr;
        }
    }

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

NS_SNP_END

#endif //__PACKET_HANDLER_H__