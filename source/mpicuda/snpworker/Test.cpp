#include <stdio.h>

#include <snp/snp.h>
USING_NS_SNP;

#include "../network/SocketAcceptor.h"
#include "../network/ProtocolHandler.h"
#include "../network/RenameMe.h"

#include "../network/DeviceGlue.h"
using namespace glue;

#ifdef WIN32
#include <pthread.h>
#endif

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
    typedef tmDevice<32> Device; // specify bitwidth
    const uint32 uiInstructionSize = sizeof(Device::tInstruction);

    uint16 uiCellSize     = Device::GetCellSize();   // int32-fields amount
    uint32 uiCellsPerPU   = 1;
    uint32 uiNumberOfPU   = 1;
    // todo: get these params via command line along with --test

    LOG_MESSAGE(1, "[test-server] Starting system [%dx%dx%d]...", uiCellSize, uiCellsPerPU, uiNumberOfPU);
    Startup(&oProtocolHandler, uiCellSize, uiCellsPerPU, uiNumberOfPU);
    LOG_MESSAGE(1, "[test-server] Final configuration [%dx%dx%d].", uiCellSize, uiCellsPerPU, uiNumberOfPU);

    // initialize data with some random data
    Device::tInstruction oInstruction;

    // address all cells at once (mask doesn't cover any of bits)
    snpBitfieldSet(oInstruction.m_oSearchMask._raw, 0);
    snpBitfieldSet(oInstruction.m_oSearchTag._raw, 0);

    // write constant values to the cells
    snpBitfieldSet(oInstruction.m_oWriteMask._raw, ~0);
    for (int32 iIntIndex = 0; iIntIndex < uiCellSize; iIntIndex++)
        oInstruction.m_oWriteData._raw[iIntIndex] = iIntIndex + 1;

    // perform instruction
    LOG_MESSAGE(1, "[test-server] Initialize device memory...");
    Exec(&oProtocolHandler, false, tOperation_Assign, oInstruction._raw, uiInstructionSize);

    // for now all cells must have the same value
    // let read them all using the first cell as flag
    oInstruction.m_oSearchMask._raw[0] = ~0;
    oInstruction.m_oSearchTag._raw[0] = 1;
    // after cell is address just reset this flag
    snpBitfieldSet(oInstruction.m_oWriteMask._raw, 0);
    oInstruction.m_oWriteMask._raw[0] = ~0;
    oInstruction.m_oWriteData._raw[0] = 0;

    // start the reading
    LOG_MESSAGE(1, "[test-server] Reading data from device...");

    uint64 iIterations = 0;
    Device::tBitfield oBitfield;
    while(Exec(&oProtocolHandler, true, tOperation_Assign, oInstruction._raw, uiInstructionSize) == true)
    {
        bool bResult = Read(&oProtocolHandler, oBitfield._raw);
        if (!bResult)
            LOG_MESSAGE(1, "[test-server] Error while reading!");

        iIterations += 1;
    }

    LOG_MESSAGE(1, "[test-server] Read %llu memory cells.", iIterations);

    // that's all
    LOG_MESSAGE(1, "[test-server] Shutting down...");
    Shutdown(&oProtocolHandler);

    LOG_MESSAGE(1, "[test-server] Thread finished.");
    return nullptr;
}
