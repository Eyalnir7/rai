BASE   = ../..
NAME   = $(shell basename `pwd`)
OUTPUT = lib$(NAME).so

JSON = 1

DEPEND = Core Algo

SRCS = $(shell find . -maxdepth 1 -name '*.cpp' )
OBJS = $(SRCS:%.cpp=%.o)
ifdef TORCH_DIR
  CXXFLAGS += -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
  LDFLAGS  += -L$(TORCH_DIR)/lib -Wl,-rpath,$(TORCH_DIR)/lib,-rpath,$(BASE)/lib
  
  # Check if the CUDA library exists in the specified directory
  HAS_CUDA := $(shell if [ -f $(TORCH_DIR)/lib/libtorch_cuda.so ]; then echo 1; else echo 0; fi)

  ifeq ($(HAS_CUDA),1)
    # Link CUDA libraries if found
    LIBS += -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10
  else
    # Fallback to CPU-only
    LIBS += -ltorch -ltorch_cpu -lc10
  endif
endif

include $(BASE)/_make/generic.mk
