#include <stdio.h>
#include <snp-cuda\snp-cuda.h>
using namespace snp;

int main()
{
	{
		// test initialization
		snpDevice<32> device1;
		// must be succeded
		snpErrorCode result1 = device1.configure(32, 32);

		snpDevice<64> device2;
		// must be failed despite of different bitwidth (GPU is busy)
		snpErrorCode result2 = device2.configure(32, 32);
		
		// as it wasn't configured the releasing must be failed
		snpErrorCode result3 = device2.end();

		// must be suceeded
		snpErrorCode result4 = device1.end();

		// initialization must be okay now
		snpErrorCode result5 = device2.configure(32, 32);
	}

	typedef snpDevice<128> Device;	

	Device device;
	// must be configured successfully as previous device was
	// released when we quitted his scope
	snpErrorCode result = device.configure(128, 32);

	Device::snpInstruction instruction;
	snpBitfieldSet(instruction.raw.bitfield, 0);
	instruction.field.singleCell = true;
	instruction.field.operation = Device::snpAssign;

	snpErrorCode result2 = device.exec(instruction);

	return 0;
}
