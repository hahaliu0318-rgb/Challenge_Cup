# 如何查看输入到输出的过程

1. 打印请求JSON：确认图像路径、问题、任务字段及双图顺序。
2. `POST /v1/routes/preview`：查看 `image_meta`、`task`、`route.reason` 和 `warnings`。此处不加载模型，不能验证模型推理。
3. `POST /v1/jobs`：保存 `job_id` 和 `status_url`；返回202代表受理，不代表完成。
4. `GET /v1/jobs/{id}`：观察 `status`、`route_worker`、时间戳与 `error`。
5. `GET /v1/workers`：各worker的字段是 `state`、`busy`、`gpus`、`pid`、`last_used_at`；`leases`是路由器内部GPU租约。GPU利用率和空闲显存是瞬时值。
6. 读取 `runtime/logs/router.log`，查看提交、worker就绪、重试、成功/失败等事件。部分worker日志只有worker名，没有任务ID；先按任务ID定位业务事件，再结合时间与worker名对应。`gateway.log`不能代替业务日志。
7. 成功后读取 `job.result`：`raw_answer`是模型返回，`answer`是格式清洗结果，`task`是任务判断，`route`是选路依据，`worker_gpus`是实际GPU。
8. 检查 `warnings`。格式修复仍失败时也可能是 `succeeded`，必须结合警告判断结果能否使用。

真实状态包括 `queued`、`classifying`、`queued_waiting_gpu`、`loading_model`、`running`、`succeeded`、`failed`、`cancelled`。worker端的 `state`主要是 `ready/stopped/disabled`，忙碌由 `busy`表示；不能期待该端点一定显示 `loading/busy`字符串。

`timing.validation_sec`是图像校验时间；`queue_sec`统计进入队列至首次运行之间的时间，当前实现可能包含加载或分类等待；`load_sec`和`inference_sec`主要来自最终worker调用。发生分类、重启或格式修复时，这几项不是互斥阶段，不能简单相加当作总耗时。`total_sec`是从服务端请求准备开始到最终结果完成的时间，不包含本地SSH连接建立时间。TTFT、逐token输出、解码速度及各次重试的完整跨度尚未独立埋点，不应由这些字段推造。

显存门槛判定不等于整卡空闲。当前分配器按候选顺序、剩余显存和内部租约选择，不保证独占外部GPU，也不根据所有外部进程建立系统级排他锁。GPU显存随后变化仍可能导致失败。
