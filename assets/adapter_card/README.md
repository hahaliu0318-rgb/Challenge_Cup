---
license: other
license_name: qwen-research
license_link: LICENSE
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
tags:
  - remote-sensing
  - image-text-to-text
  - lora
  - peft
language:
  - en
  - zh
---

# 遥感图文统一推理系统：第一阶段通用路线适配器

Built with Qwen.

本适配器包只包含队伍第一阶段训练得到的LoRA增量参数及其处理器、词表和配置，不是独立完整模型，不包含第二阶段统一基座模型。版本：`phase1-20260903-rc1`。打包与文件核验信息见包内 `RELEASE.json`；分享地址以后续公布的主项目 `assets/DOWNLOADS.md` 为准，不因文件已打包就宣称网盘上传完成。

## 文件内容

- `adapter_model.safetensors`：原始适配器主权重，148,712,776字节。
- `adapter_config.json`：LoRA结构与基座引用。
- `processor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`：训练导出的配套处理文件。
- `SHA256SUMS.txt`：发布文件核验清单。
- `OFFICIAL_DEPENDENCIES.md`、`official_dependencies_verified.json`：统一系统其他路线依赖的官方版本链接；不含其权重。
- `LICENSE`、`NOTICE`：上游研究许可及本项目修改说明。
- `RELEASE.json`：文件大小、SHA256、配置修改说明、权重结构检查和配套代码commit。

不发布训练状态、优化器、数据集、服务器路径、凭据或基座权重。服务器上的原模型未被改写。

## 下载与使用步骤

1. 从发布方提供的百度网盘分享链接下载指定版本的完整适配器包，核对包哈希。ZIP已有一层 `general_adapter/`：例如解压到 `/opt/challenge-assets/` 后，权重应直接位于 `/opt/challenge-assets/general_adapter/adapter_model.safetensors`，不要多嵌套一层同名目录。链接尚未提供时不要使用其他同名文件替代。
2. 根据 `OFFICIAL_DEPENDENCIES.md` 下载匹配的原始非AWQ通用基座，保存到 `general_base/`；不能将本适配器挂到其他模型、其他规模或AWQ版本上。
3. 在适配器目录执行 `sha256sum -c SHA256SUMS.txt`。主权重SHA256应为：

   ```text
   aaaff4f5215f243b647ccb633dabff533d9bde2d28eb4cb1da12dae2b49c399c
   ```

4. 从 [配套代码的phase1-release分支](https://github.com/hahaliu0318-rgb/Challenge_Cup/tree/phase1-release) 获取源码，按README建立环境、设置本机路径，再执行预检与启动。代码仓库目前为私有，需要发布方授予读取权限；公开适配器不自动授予私有源码访问权。
5. 先使用 `preview` 检查小图或双图路线，再用异步提交和任务查询查看推理。约10 GiB空闲显存是通用路线的调度门槛，不是所有输入都不会OOM的保证。

## 适配器结构与版本边界

LoRA秩为16，alpha为32，dropout为0.05，bias为none；作用模块包括 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。这些字段来自已核对的适配器配置，不据此补推历史学习率、epoch或训练数据规模。

发布副本将 `base_model_name_or_path` 从训练时相对路径规范化为 `Qwen/Qwen2.5-VL-3B-Instruct`，仅调整基座引用；主权重逐字节保持不变。运行时应显式选择下载好的本地非AWQ基座。

服务器观测环境为Python3.10、PyTorch2.6.0+cu124、Transformers5.12.1、PEFT0.19.1；这是发布整理时的环境观测，不等于历史训练环境，也不代表已经完成全新机器安装验收。

## 用途与验证边界

本适配器用于遥感小图描述、图像问答、计数、定位以及双图变化描述。输入以RGB遥感图像及自然语言问题为主；双图须固定前时相、后时相顺序。模型不能保证细小目标识别、精确计数或坐标可靠，不输出经验证的变化掩膜。

配套仓库提供部分历史输入输出。此次发布未重新执行完整模型评测、干净机器GPU推理、性能测量或训练防泄漏审计，因此不在本模型卡宣称新的准确率、吞吐量或完整复现成功。下载/哈希核验只证明文件一致，不证明模型回答正确。

## 许可、署名与限制

基座适用Qwen RESEARCH LICENSE AGREEMENT，全文见 `LICENSE`；不得替换为MIT/Apache等宽松许可。请遵守其研究与评估、非商业使用及再分发条款，其他用途按上游要求取得必要授权。本项目没有赋予超出上游许可的权利。

适配器由队伍微调形成，发布方已确认其公开发布权。其他路线的官方模型与代码适用各自许可。数据集不随本仓库发布，其许可和下载条件应另行遵守。遥感输出仅作研究辅助，应依据原图和独立证据复核，不作为安全关键决策的唯一依据。
