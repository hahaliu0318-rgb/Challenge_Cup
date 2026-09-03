# 适配器来源、修改与使用范围

Built with Qwen.

本项目适配器由队伍在 `Qwen/Qwen2.5-VL-3B-Instruct` 上进行LoRA微调获得，发布方已确认适配器公开发布权。公开内容仅为增量权重及必需的配置、词表和处理器文件；不提供基座镜像，也不包含训练数据和训练状态。

上游3B基座采用 **Qwen RESEARCH LICENSE AGREEMENT**，不能误标为Apache-2.0或MIT。请在使用和再分发前阅读 [许可原文](licenses/QWEN_RESEARCH_LICENSE.txt)，遵守其研究/评估、非商业用途及分发要求；其他用途应按原许可取得必要授权。本说明不扩大上游授予的权利。

需要保留的上游NOTICE：

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.

本发布版的修改说明：新增本项目LoRA训练得到的增量参数；发布副本中 `adapter_config.json` 的 `base_model_name_or_path` 从训练目录中的相对路径改为官方仓库ID。适配器权重保持原样，并附SHA256供核验。原服务器模型和配置不改写。

许可原文来源：作者固定版本 [LICENSE](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE)。许可文本仅适用于相应组件，不将本仓库所有代码、图像或其他模型统一置于该许可。
