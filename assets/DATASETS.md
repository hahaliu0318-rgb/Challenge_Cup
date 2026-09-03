# 官方数据来源

完整数据集不放入GitHub或邮件包；下载后记录所用版本、划分与样本ID。

| 数据集 | 官方入口 | 本次使用 |
|---|---|---|
| VRSBench | https://github.com/lx709/VRSBench | 验证图 `Images_val/05863_0000.png` |
| LEVIR-CC | https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset | `images/val/A/val_000001.png` 和对应B图 |
| XLRS-Bench | https://github.com/AI9Stars/XLRS-Bench | lite历史评测3080条；主页第88行样本的4096×4096原始嵌入JPEG随包提供 |
| MME-RealWorld遥感子集 | https://github.com/yfzhang114/MME-RealWorld | 主页color/0043使用 `remote_sensing/dota_v2_dota_v2_dota_v2_P10015.png` 原图；旧示例另用 `03553_Toronto.png` |

以上为数据集内部路径，不是服务器账号路径。包内5张示例图的校验值见 `examples/images_manifest.json`，三例展示的标注和真实预测见 `examples/showcase/`。XLRS数据集路径以 `.png` 结尾，但Arrow内存的是JPEG字节，导出为 `.jpg`；直接复制编码字节，不重采样、不重新编码。MME主页图为2098×2098原PNG。

旧示例 `03553_Toronto.png` 未随包复制，按官方路径下载后放入 `runtime/examples/03553_Toronto.png`，应保持原始11500×7500像素；缩放会改变路线和实验条件。它不是主页新增的color/0043图片。

MME问题必须包含全部选项；双图必须A在前B在后。不能将Arrow文件本身作为输入图像，需用数据集库解码并临时保存当前样本，待服务器任务结束后再删除；客户端断开时不得提前删除服务端仍需使用的图像。

示例归原数据提供者所有，公开再分发需核对原许可。本次仅附必要示例，不代表整个训练数据的划分和防泄漏审计已完成。
