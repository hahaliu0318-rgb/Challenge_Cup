# 依赖、版本和署名（维护者附录）

主README使用“通用、全局、细节”说明用户操作；本附录保留真实模型和项目名称以满足复现和署名要求，不能把第一阶段多模型部署表述成单一基座。

| 路线/部件 | 实际依赖 | 上游来源 |
|---|---|---|
| 通用单图及双图 | Qwen2.5-VL-3B-Instruct + 本项目Full LoRA | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |
| 高分辨率全局理解 | GeoLLaVA-8K、LongVA推理代码 | https://github.com/MiliLab/GeoLLaVA-8K |
| 高分辨率局部细节 | ZoomSearch、LLaVA-OneVision-Qwen2-7B、VisRAG-Ret | https://github.com/kiki-zyq/ZoomSearch |
| 细节路线推理实现 | LLaVA-NeXT | https://github.com/LLaVA-VL/LLaVA-NeXT |
| 全局路线视觉编码器 | CLIP ViT-L/14-336 | https://huggingface.co/openai/clip-vit-large-patch14-336 |
| 细节路线视觉编码器 | SigLIP SO400M/14-384 | https://huggingface.co/google/siglip-so400m-patch14-384 |

固定来源：全局代码 `0f45c9f28170f0d86d07a71e2b21e5e28ebeee54`；细节搜索初始源码 `d2158264f4dc4e80674788e5be5126e1399efdb1`；LLaVA-NeXT初始源码 `bce12e479bc4dfee2b9cc50c88137b01ff51bd483`。初始提交不等于服务器最终文件：细节搜索含本地双卡、图像策略、内存回收及环境兼容改动；应以依赖快照的逐文件SHA256为准。不可直接使用上游最新分支替代并宣称相同版本。

`external/`中的依赖不纳入本GitHub源码包。维护者已在本地保留服务器推理源码快照；正式上传其下载链接前必须核对完整许可及允许的分发范围。尤其不能因为仓库公开、项目名称被弱化，就删除署名或推定全部权重与数据可以自由重分发。仅写链接但遗漏本地兼容改动也不足以复现。

`integrations/`保留路由实际调用的辅助加载和评测脚本；`training/`保留通用路线的训练入口。入口存在不代表训练数据划分、采样顺序和历史超参数已完整封存。对应训练运行清单未完整采集时，不承诺从头训练得到逐位相同权重。

本仓库未替作者或队伍新设统一开源许可证。自有代码发布许可、第三方依赖许可及模型权重许可应分别确认。参赛提交也需遵守原许可。
