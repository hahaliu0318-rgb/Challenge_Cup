# 遥感图文统一推理系统 · 第一阶段

输入一张或两张遥感图像、问题和可选任务类型，系统自动选择处理路线并返回答案。用户不需要手工指定模型。

> 当前为提交整理候选版，不是已完成所有验收的正式发布。源码与示例随仓库提供，六项官方模型依赖已核对并固定版本；本项目适配器的百度网盘链接尚待补充，固定运行源码包也有待补齐，见 [大文件下载清单](assets/DOWNLOADS.md)。在补齐材料前，新用户不能仅靠本仓库复现全部真实模型推理。已有完整环境的用户可配置现有路径运行。这里只说明第一阶段，不包含第二阶段统一基座或变化掩膜能力。

## 1. 下载代码

在 Linux GPU 服务器执行（以下命令均从仓库根目录开始）：

```bash
git clone --branch phase1-release https://github.com/hahaliu0318-rgb/Challenge_Cup.git
cd Challenge_Cup
```

也可以解压提交包后进入其中的 `Challenge_Cup` 目录。仓库若为私有，需要先取得读取权限。

## 2. 准备运行材料

按 [下载清单](assets/DOWNLOADS.md) 获取适配器、固定运行源码，以及 [已核对的官方依赖](assets/OFFICIAL_DEPENDENCIES.md)。官方模型下载指定版本的完整推理文件；适配器和源码包按各自说明核验 SHA256。**不要把模型文件提交到 GitHub。**

建议目录布局：

```text
Challenge_Cup/             本仓库
  external/               依赖源码包解压到这里，Git默认忽略
/opt/challenge-assets/    自行选择的模型目录，不要求使用该路径
  general_base/
  general_adapter/
  global_model/
  detail_model/
  retrieval_model/
  detail_vision/
  global_vision/
```

首次部署按 [环境安装步骤](environments/README.md) 建立四套独立环境。若已有可用环境，直接复用，禁止把所有依赖合并安装到一个环境。

通用路线需要约10 GiB空闲显存；全局高分辨率路线需要两张各约22 GiB；细节搜索路线需要约23 GiB和9.5 GiB的双卡组合。这些是调度门槛，不是性能保证；仍需为其他进程的显存增长留出余量。

## 3. 生成本机配置

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r router/requirements-router.txt
.venv/bin/python scripts/configure.py \
  --asset-root /opt/challenge-assets \
  --env-root /opt/challenge-envs \
  --general-gpus 0 \
  --global-gpus 0,1 \
  --detail-gpus 1,2
```

上面的GPU编号只是“三卡服务器”的示例，不是固定值。先用 `nvidia-smi` 查看自己的物理GPU编号，再填写；细节路线的顺序是主推理卡在前、搜索卡在后。

为大图路线创建本地配置视图（只新增小配置和符号链接，不复制或修改权重）：

```bash
python3 scripts/prepare_model_views.py --asset-root /opt/challenge-assets
```

这是迁移时的必要步骤：视觉依赖位置可能写在模型自身的配置里，仅修改路由器YAML还不够。已有部署也可以在本地YAML直接指向已验证且视觉依赖路径正确的模型目录。

脚本生成 `router/config/router.yaml` 和 `router/config/models.yaml`。路径不同就在这两个本地文件中修改：模型位置、三套环境的Python位置、依赖源码位置、GPU候选编号、图像白名单。配置不会进入Git。

输入图像只允许位于白名单目录。默认允许本仓库的 `examples/` 和 `runtime/`；测试自己的图像时，将其服务器目录加入 `allowed_image_roots`，不要把 `/` 加入白名单。

## 4. 检查并启动

```bash
.venv/bin/python scripts/preflight.py
cd router
../.venv/bin/python -m unittest discover -s tests -v
cd ..
bash scripts/start.sh
```

检查失败时先补齐缺失文件/环境，不要忽略错误。启动只运行网关，不提前加载模型。默认仅监听 `127.0.0.1:7860`，同一套任务库只能运行一个网关进程。

本发布版启动脚本在前台运行，请保持服务器终端开启；长时间使用可在 `tmux new -s challenge-router` 中执行启动命令，再按 `Ctrl+B`、`D` 脱离。它不会更改已有服务器服务或自动重启原部署。

## 5. 从本地电脑访问

在本地PowerShell建立SSH隧道，并保持窗口开启：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 7860:127.0.0.1:7860 用户名@服务器地址 -p SSH端口
```

浏览器打开 `http://127.0.0.1:7860/docs`。服务器自身不需要SSH隧道。不要将服务端口直接开放到公网；本版本没有公网鉴权。

## 6. 选择示例并检查路由

在服务器仓库根目录执行：

```bash
python3 scripts/client.py samples
python3 scripts/client.py preview --sample small_vqa
```

示例说明和已有原始输出见 [examples/README.md](examples/README.md)。`preview`只校验图片、识别任务并给出路线，不加载模型；低置信度文本分类会推迟到真实提交时，所以预览路线可能是暂定结果。

自定义请求只需要三个字段，保存为 `request.json`：

```json
{
  "image": "/绝对服务器路径/image.png",
  "text": "What is the prominent man-made structure?",
  "task_type": "vqa"
}
```

```bash
python3 scripts/client.py preview --request request.json
```

`image`是服务器路径，不是Windows本地路径，也不是网页上传字段。双图改为 `["/before.png", "/after.png"]`，严格按前、后顺序。`task_type`可省略或为 `null`；选择题必须将全部选项写入 `text`，问题必须明确指出目标对象。

## 7. 提交并观察全过程

推荐异步提交，立即保留任务ID：

```bash
python3 scripts/client.py submit --sample small_vqa
python3 scripts/client.py watch --job-id 上一步返回的任务ID
```

第一次任务可能经历 `queued → loading_model → running → succeeded`；显存不足时停留在 `queued_waiting_gpu`。低置信度任务可能先出现 `classifying`。轮询可能看不到很短的状态，不表示缺少处理步骤。

另开一个终端查看资源与日志：

```bash
python3 scripts/client.py workers
tail -f runtime/logs/router.log
```

`router.log`记录业务与worker信息，`gateway.log`主要记录进程/HTTP访问。日志不是逐token可视化，也不保证每个内部算子都有事件。最终任务结果在 `result` 下，包括任务判断、路由原因、图像尺寸、实际GPU、原始输出、规范化答案和耗时。详见 [过程字段说明](docs/TRACE.md)。

排队任务可以取消，已运行任务不支持该取消接口：

```bash
python3 scripts/client.py cancel --job-id 任务ID
```

退出客户端轮询不会自动取消服务器任务；使用同一个任务ID可重新查询。

## 8. 同步推理、双图与大图测试

```bash
python3 scripts/client.py infer --sample small_vqa
python3 scripts/client.py preview --sample change_pair
python3 scripts/client.py submit --sample change_pair
```

同步接口会等待最终结果，不适合GPU长期不足时使用；`trace=false`只返回精简答案。大图案例按示例说明准备原图后再提交，不要为了减小文件将大图缩小为小图，否则可能改变路由。

先完成小图，再测试双图、大图细节、大图描述，避免并发挤占显存。每个worker串行处理，空闲约15分钟后释放；无合适GPU就继续排队，不跨路线降级，不停止其他用户进程。

## 9. 保存结果并停止服务

```bash
python3 scripts/client.py get --job-id 任务ID --output runtime/my-result.json
bash scripts/stop.sh
```

结果至少保留原始输入、提示词、任务ID和完整返回JSON。输出可能包含错误答案，“成功”仅代表推理流程完成，不等于回答正确。不要把运行日志、任务库和私人输入直接上传公开仓库。

## 10. 提交材料

按 [提交检查清单](docs/SUBMISSION_CHECKLIST.md) 补齐技术报告、申报表和大文件链接。只把本目录的发布文件上传仓库，不要对原服务器目录或整个研究工作区执行 `git add .`。

```bash
python3 scripts/check_release.py
python3 scripts/build_submission.py --output ../phase1_submission_DRAFT.zip
```

默认压缩包为草稿名。正式邮件包由队伍另行命名为 `申报人所在单位－申报人姓名－作品名称－联系电话.zip`，并在包内说明队伍名称。个人信息不必放入公开GitHub。

本次整理的验证范围见 [VALIDATION.md](docs/VALIDATION.md)。模型和数据来源集中在 [依赖与署名](docs/DEPENDENCIES.md) 中，不影响上述用户操作步骤。

Built with Qwen. 通用路线遵循上游研究许可，许可原文及适用范围见 [权重许可说明](assets/ADAPTER_LICENSE.md)。本仓库未替第三方代码、模型或数据重新授权。
