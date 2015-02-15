#include "DeviceGlue.h"
#include "ProtocolHandler.h"
#include "Packet.h"

NS_SNP_BEGIN

namespace glue {

bool Startup(CProtocolHandler *pHandler, uint16 &uiCellSize, uint32 &uiCellsPerPU, uint32 &uiNumberOfPU)
{
    tPacket oRequest;
    oRequest.m_eType = tPacket::tType_RequestStartup;
    oRequest.m_oData.asRequestStartup.m_uiCellSize = uiCellSize;
    oRequest.m_oData.asRequestStartup.m_uiCellsPerPU = uiCellsPerPU;
    oRequest.m_oData.asRequestStartup.m_uiNumberOfPU = uiNumberOfPU;
    pHandler->Write(&oRequest);

    tPacket oResponse = pHandler->Read();
    assert(oResponse.m_eType == tPacket::tType_ResponseStartup);
    if (oResponse.m_oData.asResponseStartup.m_bResult)
    {
        uiCellSize = oResponse.m_oData.asResponseStartup.m_uiCellSize;
        uiCellsPerPU = oResponse.m_oData.asResponseStartup.m_uiCellsPerPU;
        uiNumberOfPU = oResponse.m_oData.asResponseStartup.m_uiNumberOfPU;
    }
    return oResponse.m_oData.asResponseStartup.m_bResult;
}

bool Shutdown(CProtocolHandler *pHandler)
{
    tPacket oRequest;
    oRequest.m_eType = tPacket::tType_RequestShutdown;
    pHandler->Write(&oRequest);

    tPacket oResponse = pHandler->Read();
    assert(oResponse.m_eType == tPacket::tType_ResponseShutdown);
    return oResponse.m_oData.asResponseShutdown.m_bResult;
}

bool Exec(CProtocolHandler *pHandler, bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction, uint32 uiInstructionSize)
{
    tPacket oRequest;
    oRequest.m_eType = tPacket::tType_RequestExec;
    oRequest.m_oData.asRequestExec.m_bSingleCell = bSingleCell;
    oRequest.m_oData.asRequestExec.m_eOperation = eOperation;
    oRequest.m_oDynamicData.resize(uiInstructionSize);
    memcpy(&oRequest.m_oDynamicData.front(), pInstruction, uiInstructionSize);
    pHandler->Write(&oRequest);

    tPacket oResponse = pHandler->Read();
    assert(oResponse.m_eType == tPacket::tType_ResponseExec);
    return oResponse.m_oData.asResponseExec.m_bResult;
}

bool Read(CProtocolHandler *pHandler, uint32 *pBitfield)
{
    tPacket oRequest;
    oRequest.m_eType = tPacket::tType_RequestRead;
    pHandler->Write(&oRequest);

    tPacket oResponse = pHandler->Read();
    assert(oResponse.m_eType == tPacket::tType_ResponseRead);

    if (oResponse.m_oData.asResponseRead.m_bResult)
    {
        // copy output data
        assert(oResponse.m_oDynamicData.size() > 0);
        memcpy(pBitfield, &oResponse.m_oDynamicData.front(), oResponse.m_oDynamicData.size());
    }
    return oResponse.m_oData.asResponseRead.m_bResult;
}

}

NS_SNP_END
