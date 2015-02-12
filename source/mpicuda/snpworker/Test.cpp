#include <stdio.h>

#include <snp/snp.h>
USING_NS_SNP;

#include "../network/SocketAcceptor.h"
#include "../network/ProtocolHandler.h"

NS_SNP_BEGIN

class CServerProtocolHandler: public CProtocolHandler
{
public:
    CServerProtocolHandler()
        : m_pPacket(NULL)
    {}

    virtual ~CServerProtocolHandler()
    {
        if (m_pPacket)
            delete m_pPacket;
    }

    inline tPacket * GetPacket() const
    {
        return m_pPacket;
    }
    
    inline void NextPacket()
    {
        if (m_pPacket)
        {
            delete m_pPacket;
            m_pPacket = NULL;
        }
    }

private:
    inline void Execute()
    {
        if (!m_pPacket)
            m_pPacket = GrabPacket();
    }

    tPacket *m_pPacket;
};

NS_SNP_END

static bool Startup(CServerProtocolHandler *pHandler, uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU);
static bool Shutdown(CServerProtocolHandler *pHandler);
static bool Exec(CServerProtocolHandler *pHandler, bool bSingleCell, snpOperation eOperation, const uint32 * const pInstruction);
static bool Read(CServerProtocolHandler *pHandler, uint32 *pBitfield);

static tPacket * WaitPacket(CProtocolHandler *pHandler);

extern "C" void * ThreadServerF(void *pArg)
{
    LOG_MESSAGE(1, "[test-server] Thread started...");
    LOG_MESSAGE(1, "[test-server] Connection...");

    CSocketAcceptor oSocketAcceptor;
    oSocketAcceptor.AcceptConnections();

    LOG_MESSAGE(1, "[test-server] Connection established...");
    int iSocketFD = oSocketAcceptor.GetAcceptedSocket();

    snp::CServerProtocolHandler oProtocolHandler;
    oProtocolHandler.InitSocket(iSocketFD);
    LOG_MESSAGE(1, "[test-server] Socket created.");
    
    //
    // Start test sequence
    typedef snpDevice<1024> Device; // specify bitwidth

    const uint16 uiCellSize     = Device::getCellSize();   // int32-fields amount
    const uint32 uiCellsPerPU   = 128;
    const uint32 uiNumberOfPU   = 10000;
    // todo: get these params via command line along with --test

    LOG_MESSAGE(1, "[test-server] Starting system (uiCellSize=%d, uiCellsPerPU=%d, uiNumberOfPU=%d)...", uiCellSize, uiCellsPerPU, uiNumberOfPU);
    Startup(&oProtocolHandler, uiCellSize, uiCellsPerPU, uiNumberOfPU);

    // initialize data with some random data
    Device::snpInstruction oInstruction;

    // address all cells at once (mask doesn't cover any of bits)
    snpBitfieldSet(oInstruction.field.addressMask.raw, 0);
    snpBitfieldSet(oInstruction.field.addressData.raw, 0);

    // write constant values to the cells
    snpBitfieldSet(oInstruction.field.dataMask.raw, ~0);
    for (int32 iIntIndex = 0; iIntIndex < uiCellSize; iIntIndex++)
        oInstruction.field.dataData.raw[iIntIndex] = iIntIndex;

    // perform instruction
    LOG_MESSAGE(1, "[test-server] Initialize device memory...");
    Exec(&oProtocolHandler, false, snpAssign, oInstruction.raw);

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
    while(Exec(&oProtocolHandler, true, snpAssign, oInstruction.raw) == true)
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

static bool Startup(CServerProtocolHandler *pHandler, uint16 uiCellSize, uint32 uiCellsPerPU, uint32 uiNumberOfPU)
{
    //tPacket *pPacket = new tPacket();
    //pPacket->m_eType = tPacket::tType_Startup;
    //oProtocolHandler.AddOutgoingPacket(pPacket);

    tPacket *pResponse = WaitPacket(pHandler);
    bool bResult = false; // todo: handle response here
    pHandler->NextPacket();
}

static bool Shutdown(CServerProtocolHandler *pHandler)
{
    tPacket *pResponse = WaitPacket(pHandler);
    bool bResult = false; // todo: handle response here
    pHandler->NextPacket();
}

static bool Exec(CServerProtocolHandler *pHandler, bool bSingleCell, snpOperation eOperation, const uint32 * const pInstruction)
{
    tPacket *pResponse = WaitPacket(pHandler);
    bool bResult = false; // todo: handle response here
    pHandler->NextPacket();
}

static bool Read(CServerProtocolHandler *pHandler, uint32 *pBitfield)
{
    tPacket *pResponse = WaitPacket(pHandler);
    bool bResult = false; // todo: handle response here
    pHandler->NextPacket();
}

static tPacket * WaitPacket(CServerProtocolHandler *pHandler)
{
    if (!pHandler) return NULL;

    tPacket *pPacket = NULL;
    while(!(pPacket = pHandler->GetPacket()))
    {
        msleep(10);
        pHandler->Tick();
    }
    return pPacket;
}
