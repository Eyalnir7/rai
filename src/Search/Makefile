BASE   = ../..
NAME   = $(shell basename `pwd`)
OUTPUT = lib$(NAME).so

JSON = 1

DEPEND = Core Algo

SRCS = $(shell find . -maxdepth 1 -name '*.cpp' )
OBJS = $(SRCS:%.cpp=%.o)
TORCH_DIR = /home/eyal/git/libtorch-shared-with-deps-2.9.1+cpu/libtorch
CPATHS += $(TORCH_DIR)/include $(TORCH_DIR)/include/torch/csrc/api/include
LPATHS += $(TORCH_DIR)/lib
LIBS   += -ltorch -ltorch_cpu -lc10

include $(BASE)/_make/generic.mk
