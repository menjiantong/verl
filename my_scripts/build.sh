#!/bin/bash
set -eo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
EXTRA_INDEX="--extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn"

echo " ============= 1. build vllm (empty device for Ascend) ============= "
cd vllm
pip install -r requirements/common.txt
VLLM_TARGET_DEVICE=empty pip install  -v -e .
cd ..

echo " ============= 2. build vllm-ascend ============= "
cd vllm-ascend
pip install -r requirements.txt $EXTRA_INDEX
export COMPILE_CUSTOM_KERNELS=1
export SOC_VERSION=ascend910_9392
pip install  -v -e . $EXTRA_INDEX
cd ..

echo " ============= 3. build FSDPTurbo ============= "
cd FSDPTurbo
pip install  -e . $EXTRA_INDEX
cd ..

echo " ============= 4. build verl ============= "
cd verl
pip install -r requirements-npu.txt $EXTRA_INDEX
pip install  -v -e .
cd ..

echo " ============= 5. pin transformers & tokenizers ============= "
pip install transformers==5.21.0 tokenizers==0.23.3