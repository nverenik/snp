#include <stdio.h>
#include <stdlib.h>

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
	snpErrorCode result = device.configure(4, 4);

	Device::snpInstruction instruction;

	// address all cells at once (mask doesn't cover any of bits)
	snpBitfieldSet(instruction.field.addressMask.raw, 0);
	snpBitfieldSet(instruction.field.addressData.raw, 0);
	// write '1'-'2'-'3' to the first three integers and random value to the last one
	snpBitfieldSet(instruction.field.dataMask.raw, ~0);
	instruction.field.dataData.raw[0] = 1;
	instruction.field.dataData.raw[1] = 2;
	instruction.field.dataData.raw[2] = 3;
	instruction.field.dataData.raw[3] = 4;
	// perform instruction
	device.exec(false, snpAssign, instruction);
	device.dump();

	// for now all cells must have the same values outside of the last integer
	// let read them all using the first cell as flag
	instruction.field.addressMask.raw[0] = ~0;
	instruction.field.addressData.raw[0] = 1;
	// after cell is address just reset this flag
	snpBitfieldSet(instruction.field.dataMask.raw, 0);
	instruction.field.dataMask.raw[0] = ~0;
	instruction.field.dataData.raw[0] = 0;

	Device::snpBitfield bitfield;
	while(device.exec(true, snpAssign, instruction) == true)
	{
		printf("\n");
		device.dump();

		// result must be true in any case
		bool result = device.read(bitfield);

		printf("\n");
		for (uint32 index = 0; index < device.getCellSize(); index++)
		{
			printf("%d ", bitfield.raw[index]);
		}
		printf("\n");
	}

	return 0;
}
