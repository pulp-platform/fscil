# User Test
#------------------------------------

APP              = test
APP_SRCS        += prototype.c
APP_SRCS        += network.c
APP_SRCS        += uart_controller.c
APP_SRCS        += uart_buffer.c
APP_SRCS        += mem.c
APP_SRCS        += main.c
APP_INC         +=
APP_CFLAGS      += #-DUART_FLOW_CONTROL_EMU


APP_CFLAGS += -DNUM_CORES=$(CORE) -Iinc -O2 -fno-indirect-inlining -flto -w
APP_LDFLAGS += -lm -Wl,--print-memory-usage -flto
FLASH_TYPE ?= HYPERFLASH
RAM_TYPE ?= HYPERRAM

ifeq '$(FLASH_TYPE)' 'MRAM'
READFS_FLASH = target/chip/soc/mram
endif

APP_CFLAGS += -DFLASH_TYPE=$(FLASH_TYPE) -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS

include $(GAP_SDK_HOME)/utils/rules/pmsis_rules.mk
