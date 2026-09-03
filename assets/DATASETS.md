# 官方数据来源

完整数据集不放入GitHub或邮件包；下载后记录所用版本、划分与样本ID。

| 数据集 | 官方入口 | 本次使用 |
|---|---|---|
| VRSBench | https://github.com/lx709/VRSBench | 验证图 `Images_val/05863_0000.png` |
| LEVIR-CC | https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset | `images/val/A/val_000001.png` 和对应B图 |
| XLRS-Bench | https://github.com/AI9Stars/XLRS-Bench | lite历史评测3080条；部分图像嵌入Arrow |
| MME-RealWorld遥感子集 | https://github.com/yfzhang114/MME-RealWorld | `remote_sensing/03553_Toronto.png` |

以上为数据集内部路径，不是服务器账号路径。包内示例校验值见 `examples/images_manifest.json`。大图未随包复制，按官方路径下载后放入 `runtime/examples/03553_Toronto.png`，应保持原始11500×7500像素；缩放会改变路线和实验条件。

MME问题必须包含全部选项；双图必须A在前B在后。不能将Arrow文件本身作为输入图像，需用数据集库解码并临时保存当前样本，待服务器任务结束后再删除；客户端断开时不得提前删除服务端仍需使用的图像。

示例归原数据提供者所有，公开再分发需核对原许可。本次仅附必要示例，不代表整个训练数据的划分和防泄漏审计已完成。
