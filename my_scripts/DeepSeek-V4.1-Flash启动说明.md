# DeepSeek-V4.1-Flash启动说明

## 1. 单机A3


拉取镜像：
```
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-ubuntu22.04-py3.11
```

下载代码
```
git clone -b dev_dsv41_1027  https://gitcode.com/menjiantong/FSDPTurbo
git clone -b from_9a38d36_1030 https://github.com/menjiantong/vllm-ascend/
git clone -b from_a98247a_1030 https://github.com/menjiantong/vllm/
git clone -b dev_dsv41_rebase1030_cp_engram https://github.com/menjiantong/verl

cd FSDPTurbo
git reset --hard cc3cd1eba9466a03be0f85922eae2e644bda0b5a
cd ..

cd vllm-ascend/
git reset --hard b0901f799fa1a178254304d44269fe352dbce315
cd ..

cd vllm
git reset --hard fd46ca5adaee11331a4c2f1b74483fef491f03cf
cd ..

cd verl 
git reset --hard 6302c3f682cf159299b90bdeeea22f1286d70e11
cd ..
```

安装环境
```
bash build.sh
bash 1110_run_dsv41_188_longrun_v1.sh
```