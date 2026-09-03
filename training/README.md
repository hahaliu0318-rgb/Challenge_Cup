# 训练复现范围

1. 从官方数据入口取得原始数据，准备训练/验证JSONL并核对图像ID不交叉。
2. 使用通用路线独立环境，先查看入口参数：

```bash
/opt/challenge-envs/general/bin/python integrations/qwen/train_qwen25vl_lora.py --help
```

3. 根据原训练运行清单填写基座、训练文件、验证文件、输出目录、随机种子、训练轮数、学习率、batch/累积步数、LoRA配置等，再训练。历史运行清单尚未完整封存，所以本仓库不把脚本默认值当作本项目已采用的超参数。
4. 若目标只是推理复现，直接下载对应版本适配器，不需要重新训练。
5. 评测入口：`integrations/qwen/eval_qwen25vl_full_meta.py --help`、`integrations/geollava/xlrs_lite_eval.py --help`；细节路线批评测入口随固定依赖源码包提供。按同一数据划分和提示词运行，再保存输入清单、预测、指标、配置、环境及校验值。

尚待补充：完整训练数据处理链的版本、实际训练启动命令、随机种子与运行状态文件。未补齐前，只能声称提供推理权重版本与训练入口，不应声称训练全流程逐位可复现。
