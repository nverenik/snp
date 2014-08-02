#include <stdio.h>
#include <snp-cuda\snp-cuda.h>
using namespace snp;

int main()
{
	snpDevice<128> device1;
	snpErrorCode result1 = device1.configure(32, 32);
	//device1.end();

	snpDevice<128> device2;
	snpErrorCode result2 = device2.configure(32, 32);

	return 0;
}
