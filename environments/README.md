# 独立环境安装步骤

以下是新机器安装起点和服务器2026-09-03观察到的关键版本；不是已在干净机器完整重建通过的锁文件。全部依赖树、二进制轮子可用性、CUDA驱动兼容仍需复测。不要在已有环境直接执行升级命令。

## 1. 创建环境

安装Linux、NVIDIA驱动及Conda，选择自己有写权限的目录：

```bash
conda create -y -p /opt/challenge-envs/general python=3.10 pip
conda create -y -p /opt/challenge-envs/global python=3.11 pip
conda create -y -p /opt/challenge-envs/detail python=3.10 pip
```

网关用主README中的 `.venv`。更换前缀时同步修改配置。

## 2. 通用路线

```bash
/opt/challenge-envs/general/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
/opt/challenge-envs/general/bin/python -m pip install transformers==5.12.1 accelerate==1.14.0 peft==0.19.1 qwen-vl-utils==0.0.14 numpy==2.2.6 Pillow==12.2.0
```

这些版本取自当前已有环境。若包索引没有相同版本，不能悄悄用最新版本替代；需提供原环境对应wheel/安装来源并重新验证。本阶段通用推理不使用4bit/AWQ；bitsandbytes并非该推理配置的必要项。

## 3. 全局大图路线

先按下载清单准备 `external/geollava8k/longva`：

```bash
/opt/challenge-envs/global/bin/python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
/opt/challenge-envs/global/bin/python -m pip install transformers==4.43.4 accelerate==0.34.2 numpy==1.26.4 Pillow==11.0.0 datasets==5.0.1 einops sentencepiece protobuf packaging ninja
/opt/challenge-envs/global/bin/python -m pip install flash-attn==2.7.3 --no-build-isolation
/opt/challenge-envs/global/bin/python -m pip install -e external/geollava8k/longva --no-deps
```

FlashAttention需要匹配的编译工具链/CUDA Toolkit或兼容wheel。未锁定的补充依赖要在干净机器验证后写入完整锁文件；当前 `*.observed.json` 只记录关键包。

## 4. 细节大图路线

准备已包含兼容改动的 `external/zoomsearch` 和 `external/llava-next`：

```bash
/opt/challenge-envs/detail/bin/python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
/opt/challenge-envs/detail/bin/python -m pip install -r environments/detail-requirements.txt
/opt/challenge-envs/detail/bin/python -m pip install -e external/llava-next --no-deps
```

不同路线对Transformers等依赖的要求不同，不能共用一套环境。细节路线搜索卡需支持BF16；最终GPU编号和内存门槛在本地YAML中设置。

## 5. 校验并返回主README

```bash
/opt/challenge-envs/general/bin/python -m pip check
/opt/challenge-envs/global/bin/python -m pip check
/opt/challenge-envs/detail/bin/python -m pip check
.venv/bin/python scripts/preflight.py --imports
```

`--imports`只做模块导入，不加载权重。路径、分片检查和导入通过也不等于GPU实测完成，仍需各路线至少一个真实请求。
