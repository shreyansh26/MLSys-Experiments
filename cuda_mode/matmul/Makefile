.PHONY: all build_fp32 build_fp16 debug clean profile bench cuobjdump

CMAKE := cmake

BUILD_DIR := build
BENCHMARK_DIR := benchmark_results

PRECISION ?= FP32

all: build_fp32

build_fp32:
	@mkdir -p $(BUILD_DIR)/fp32
	@cd $(BUILD_DIR)/fp32 && $(CMAKE) -DCMAKE_BUILD_TYPE=Release -DBUILD_PRECISION=FP32 ../../
	@$(MAKE) -C $(BUILD_DIR)/fp32

build_fp16:
	@mkdir -p $(BUILD_DIR)/fp16
	@cd $(BUILD_DIR)/fp16 && $(CMAKE) -DCMAKE_BUILD_TYPE=Release -DBUILD_PRECISION=FP16 ../../
	@$(MAKE) -C $(BUILD_DIR)/fp16

debug:
	@mkdir -p $(BUILD_DIR)/$(PRECISION)/debug
	@cd $(BUILD_DIR)/$(PRECISION)/debug && $(CMAKE) -DCMAKE_BUILD_TYPE=Debug -DBUILD_PRECISION=$(PRECISION) ../../
	@$(MAKE) -C $(BUILD_DIR)/$(PRECISION)/debug

clean:
	@rm -rf $(BUILD_DIR) matrixValidationFailure.txt

FUNCTION := $$(cuobjdump -symbols build/sgemm | grep -i Warptiling | awk '{print $$NF}')

cuobjdump: build
	@cuobjdump -arch sm_86 -sass -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.sass
	@cuobjdump -arch sm_86 -ptx -fun $(FUNCTION) build/sgemm | c++filt > build/cuobjdump.ptx

# Usage: make profile KERNEL=<integer> PREFIX=<optional string>
profile: build
	@ncu --set full --export $(BENCHMARK_DIR)/$(PREFIX)kernel_$(KERNEL) --force-overwrite $(BUILD_DIR)/sgemm $(KERNEL)

bench: build
	@bash gen_benchmark_results.sh