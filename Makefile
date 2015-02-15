################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /opt/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# MPI check and binaries
MPICXX           = $(shell which mpicxx)
ifeq ($(MPICXX),)
      $(error MPI not found, not building simpleMPI.)
endif

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
EXTRA_CCFLAGS   ?= -std=c++0x


# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)


# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      NVCCFLAGS += -lineinfo
      TARGET    := release

endif

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

# Project specific includes
INCLUDES += -Iinclude/ \
            -Iexternal/tclap-1.2.1/include/ \
            -I$(CUDA_INC_PATH)


EXECUTABLE = snpworker
SNPWORKER_ROOT = source/mpicuda/snpworker

SOURCES = $(SNPWORKER_ROOT)/Kernel.cu \
          $(SNPWORKER_ROOT)/Main.cpp \
          $(SNPWORKER_ROOT)/Test.cpp \
          $(SNPWORKER_ROOT)/Worker.cpp \
          $(SNPWORKER_ROOT)/../network/DeviceGlue.cpp \
          $(SNPWORKER_ROOT)/../network/Packet.cpp \
          $(SNPWORKER_ROOT)/../network/ProtocolHandler.cpp \
          $(SNPWORKER_ROOT)/../network/RenameMe.cpp \
          $(SNPWORKER_ROOT)/../network/Socket.cpp \
          $(SNPWORKER_ROOT)/../network/SocketAcceptor.cpp

OBJECTS = $(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SOURCES)))

.SUFFIXES: .cpp .cu .o

# Target rules
all: build

build: clean $(SOURCES) $(EXECUTABLE)
	make clean

$(EXECUTABLE): $(OBJECTS)
	$(MPICXX) -o $@ $+ $(LDFLAGS) $(EXTRA_LDFLAGS)

.cpp.o:
	$(MPICXX) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -c -o $@ $<

.cu.o:
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

run: build
	./snpworker

clean:
	rm -f *.o *.cudafe* *cpp*i *cubin *cu.cpp *fatbin* *linkinfo *.ptx *hashinfo *hash *.module_id
