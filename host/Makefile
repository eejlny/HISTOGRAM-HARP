# Most logic is in regtest/util/Makefile.host.
# Setting exe-specific variables here.
# Assuming source files are in 'src' sub-dir, include files are
# in 'inc' sub-dir.

TARGET = host

SRCS = AOCL_Utils.cpp \
       histogram.cpp

USES_NVIDIA = 0
USES_ACL_HOST_UTILS = 1
       
# Profiling
ifeq ($(PROFILE),1)
CPPFLAGS += -DGPU_PROFILING
endif

ifeq ($(DUMP),1)
CPPFLAGS += -DDUMP_INPUTS_OUTPUTS
endif

# can also set
#   PLATFORM = ACL|NVIDIA (default ACL)
#   DEBUG    = 0|1 (default 0)
#   LIBS
#   CPPFLAGS
#   LINKFLAGS

# Top-level regtest dir.
REG_LOCAL_ROOT_DIR_PATH ?= ../../..
include $(REG_LOCAL_ROOT_DIR_PATH)/util/Makefile.host

