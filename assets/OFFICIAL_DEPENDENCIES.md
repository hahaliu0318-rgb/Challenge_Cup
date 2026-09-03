# 已核对的官方模型依赖

本次按用户选择仅上传自训练适配器，其他模型引用作者官方仓库，不复制到本账号。以下链接固定到2026-09-03核验的commit；可访问性和配置已经检查，未完成服务器全权重逐字节比对，不能据此声称严格等价。

| 安装目录 | 官方固定版本 | 配置与服务器的对比 |
|---|---|---|
| `general_base/` | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3) | 配置JSON一致 |
| `global_model/` | [initiacms/GeoLLaVA-8K](https://huggingface.co/initiacms/GeoLLaVA-8K/tree/1000d8f7b37cd7e5752779f488635496142f5188) | 配置JSON一致 |
| `global_vision/` | [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/ce19dc912ca5cd21c8a653c79e251e808ccabcd1) | 本地配置未采集，待比对 |
| `detail_model/` | [lmms-lab/llava-onevision-qwen2-7b-ov](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov/tree/0b07bf7565e244cf4f39982249eafe8cd799d6dd) | 差异字段：mm_vision_tower |
| `detail_vision/` | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384/tree/9fdffc58afc957d1a03a25b10dba0329ab15c2a3) | 配置JSON一致 |
| `retrieval_model/` | [openbmb/VisRAG-Ret](https://huggingface.co/openbmb/VisRAG-Ret/tree/95ef596df871b606167cb7e4b7215caf1bfdf761) | 配置JSON一致 |

细节主模型只差视觉依赖的本地路径字段，使用README中的模型配置视图步骤重映射；原权重不改写。完整的revision、配置哈希及官方权重LFS哈希在 `official_dependencies_verified.json`。

通用基座的官方魔搭入口：[Qwen2.5-VL-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct)。该入口已确认可访问，但本次固定revision清单采用作者Hugging Face仓库；未经文件核对不要把两平台版本默认为相同。

基座须使用原始非AWQ版本。下载时保留词表、处理器、配置、权重索引及所有权重分片；不要只下载一个safetensors文件。

Built with Qwen. 通用路线基座适用 [Qwen RESEARCH LICENSE AGREEMENT](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE)，本项目没有把它重新许可为Apache/MIT；分发包保留对应LICENSE和NOTICE。其他依赖分别遵守上游许可。
