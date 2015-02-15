#include <stdio.h>

#include <snp/snp.h>
USING_NS_SNP;

#include "../network/SocketAcceptor.h"
#include "../network/ProtocolHandler.h"
#include "../network/RenameMe.h"

static bool Startup(CProtocolHandler *pHandler, uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU);
static bool Shutdown(CProtocolHandler *pHandler);
static bool Exec(CProtocolHandler *pHandler, bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction, uint32 uiInstructionSize);
static bool Read(CProtocolHandler *pHandler, uint32 *pBitfield);

extern "C" void * ThreadServerF(void *pArg)
{
    LOG_MESSAGE(1, "[test-server] Thread started...");
    LOG_MESSAGE(1, "[test-server] Connection...");

    CSocketAcceptor oSocketAcceptor;
    oSocketAcceptor.AcceptConnections();

    LOG_MESSAGE(1, "[test-server] Connection established...");
    int iSocketFD = oSocketAcceptor.GetAcceptedSocket();

    CProtocolHandler oProtocolHandler(iSocketFD);
    LOG_MESSAGE(1, "[test-server] Socket created.");
    
    //
    // Start test sequence
    typedef snpDevice<1024> Device; // specify bitwidth
    const uint32 uiInstructionSize = sizeof(Device::snpInstruction);

    const uint16 uiCellSize     = Device::getCellSize();   // int32-fields amount
    const uint32 uiCellsPerPU   = 128;
    const uint32 uiNumberOfPU   = 10000;
    // todo: get these params via command line along with --test

    LOG_MESSAGE(1, "[test-server] Starting system [%dx%dx%d]...", uiCellSize, uiCellsPerPU, uiNumberOfPU);
    Startup(&oProtocolHandler, uiCellSize, uiCellsPerPU, uiNumberOfPU);

    // initialize data with some random data
    Device::snpInstruction oInstruction;

    // address all cells at once (mask doesn't cover any of bits)
    snpBitfieldSet(oInstruction.field.addressMask.raw, 0);
    snpBitfieldSet(oInstruction.field.addressData.raw, 0);

    // write constant values to the cells
    snpBitfieldSet(oInstruction.field.dataMask.raw, ~0);
    for (int32 iIntIndex = 0; iIntIndex < uiCellSize; iIntIndex++)
        oInstruction.field.dataData.raw[iIntIndex] = iIntIndex + 1;

    // perform instruction
    LOG_MESSAGE(1, "[test-server] Initialize device memory...");
    Exec(&oProtocolHandler, false, tOperation_Assign, oInstruction.raw, uiInstructionSize);

    // for now all cells must have the same value
    // let read them all using the first cell as flag
    oInstruction.field.addressMask.raw[0] = ~0;
    oInstruction.field.addressData.raw[0] = 1;
    // after cell is address just reset this flag
    snpBitfieldSet(oInstruction.field.dataMask.raw, 0);
    oInstruction.field.dataMask.raw[0] = ~0;
    oInstruction.field.dataData.raw[0] = 0;

    // start the reading
    LOG_MESSAGE(1, "[test-server] Reading data from device...");

    uint64 iIterations = 0;
    Device::snpBitfield oBitfield;
    while(Exec(&oProtocolHandler, true, tOperation_Assign, oInstruction.raw, uiInstructionSize) == true)
    {
        bool bResult = Read(&oProtocolHandler, oBitfield.raw);
        if (!bResult)
            LOG_MESSAGE(1, "[test-server] Error while reading!");

        iIterations += 1;
    }

    LOG_MESSAGE(1, "[test-server] Read %llu memory cells.", iIterations);

    // that's all
    LOG_MESSAGE(1, "[test-server] Shutting down...");
    Shutdown(&oProtocolHandler);

    LOG_MESSAGE(1, "[test-server] Thread finished.");
    return NULL;
}

static bool Startup(CProtocolHandler *pHandler, uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU)
{
    tPacket oRequest;
    oRequest.m_eType = tPacket::tType_RequestStartup;
    oRequest.m_oData.asRequestStartup.m_uiCellSize = uiCellSize;
    oRequest.m_oData.asRequestStartup.m_uiCellsPerPU = uiCellsPerPU;
    oRequest.m_oData.asRequestStartup.m_uiNumberOfPU = uiNumberOfPU;
    pHandler->Write(&oRequest);

    tPacket oResponse = pHandler->Read();
    assert(oResponse.m_eType == tPacket::tType_ResponseStartup);
    return oResponse.m_oData.asResponseStartup.m_bResult;
}

static bool Shutdown(CProtocolHandler *pHandler)
{
    tPacket oRequest;
    oRequest.m_eType = tPacket::tType_RequestShutdown;
    pHandler->Write(&oRequest);

    tPacket oResponse = pHandler->Read();
    assert(oResponse.m_eType == tPacket::tType_ResponseShutdown);
    return oResponse.m_oData.asResponseShutdown.m_bResult;
}

static bool Exec(CProtocolHandler *pHandler, bool bSingleCell, tOperation eOperation, const uint32 * const pInstruction, uint32 uiInstructionSize)
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

static bool Read(CProtocolHandler *pHandler, uint32 *pBitfield)
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
