# =============================================================================
# HYDRA - Makefile
# =============================================================================
# Standalone — ne dépend pas de Barracuda/Cyclope
# Architectures : sm_120 (RTX 5060 Blackwell) + fallback sm_89 (RTX 4090)
# =============================================================================

NVCC       = nvcc
CXX        = g++
TARGET     = Hydra

# Architecture GPU cible (sm_120 = RTX 5060/5090 Blackwell)
# Modifier ici si nécessaire : sm_89 = RTX 4090, sm_86 = RTX 3090
GENCODE    = -gencode arch=compute_120,code=sm_120 \
             -gencode arch=compute_120,code=compute_120

# Flags de compilation
NVCC_FLAGS = $(GENCODE) \
             -O3 \
             -Xcompiler -O2,-march=native \
             -std=c++17 \
             --extended-lambda \
			 -Xptxas -v

# OpenSSL (pour précomputation ECC CPU)
OPENSSL_FLAGS = -lssl -lcrypto

# Includes
INCLUDES   = -I.

# Sources
SRCS       = Hydra.cu

# =============================================================================
all: $(TARGET)

$(TARGET): $(SRCS) ECC.h Hash.cuh HydraCommon.h Gray.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRCS) -o $(TARGET) $(OPENSSL_FLAGS)
	@echo "Build OK : ./$(TARGET)"

# Pour RTX 4090 / RTX 3080
sm89:
	$(NVCC) $(NVCC_FLAGS:-gencode arch=compute_120,code=sm_120=-gencode arch=compute_89,code=sm_89) \
	        $(INCLUDES) $(SRCS) -o $(TARGET) $(OPENSSL_FLAGS)

# Pour RTX 3090 / RTX 3080
sm86:
	$(NVCC) $(NVCC_FLAGS:-gencode arch=compute_120,code=sm_120=-gencode arch=compute_86,code=sm_86) \
	        $(INCLUDES) $(SRCS) -o $(TARGET) $(OPENSSL_FLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean sm89 sm86
