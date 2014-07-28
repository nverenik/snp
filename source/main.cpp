#include <stdio.h>
#include <snp-cuda\snp-cuda.h>

int main()
{
	snp::snpDevice<128> device1;
	bool result1 = device1.configure(32, 32);
	//device1.end();

	snp::snpDevice<128> device2;
	bool result2 = device2.configure(32, 32);

	return 0;
}
