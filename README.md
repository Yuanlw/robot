# 家庭机器人

第一阶段目标：识别已录入的家庭成员，听懂中文、简短回答、自然朗读，并在确认后按时提醒。当前仍是电脑端原型，尚未完成完整的家庭场景验收。

## 本地运行

推荐 Python 3.12（中文语音依赖当前不支持 Python 3.13）。

```bash
cd /Users/yuanlw/robot
python3.12 -m venv .venv-voice
.venv-voice/bin/pip install -r requirements-voice.txt
.venv-voice/bin/python scripts/download_voice.py
# 首次运行：复制 .env.example 为 .env，并在本机填写密钥；已有 .env 不要覆盖。
./run.sh
```

打开 http://127.0.0.1:8000 。`run.sh` 优先使用 `.venv-voice`，不存在时使用原 `.venv`。模型文件约 362 MiB，位于忽略提交的 `data/tts/`。

隐私边界（默认配置）：对话文字、录音识别与云端音色都会发送到阿里百炼（境内端点），照片与人脸特征仍只在本机处理。希望语音数据不出本机时：`ROBOT_STT_BACKEND=browser` 改回浏览器识别，音色在设置页切换到 Kokoro/Argmax 本地声线（对话文字本身仍走百炼模型）。

只有基础依赖也可以运行：安装 `requirements.txt`，使用 macOS 的 Tingting 中文系统音色；系统语音不可用时，网页尝试浏览器安装的中文音色，并显示失败信息。浏览器可能要求用户点击播放。

## 对话与语音

- 默认由本机 DeepSeek Harness 工具运行 Agent，模型渠道为阿里百炼：复用 `.env` 中的 `ROBOT_LLM_API_KEY`（百炼 key）、`ROBOT_LLM_BASE_URL`（百炼 OpenAI 兼容端点）和 `ROBOT_LLM_MODEL`（默认 `qwen3.8-flash`），关闭思考模式。需先安装可用的 `/usr/local/bin/dsh`。
- Agent 只开放时间、天气、提醒草稿、确认、查询和取消工具，未开放终端或文件操作。网页仍负责语音、人脸与提醒播放。
- 页面显示工具执行进度。当前 Harness SDK 返回完整回答后才显示文字并按句朗读，不是逐 token 输出；可以取消回答、停止播放和重播。服务等待上限默认 60 秒，网页上限 75 秒。
- Agent 进程常驻复用：同一会话连续对话不再每轮冷启动（实测纯聊天轮从约 1.8~2.3 秒降到约 0.8~1.0 秒），空闲 5 分钟自动回收，出错或取消时立即回收并下轮重建。`ROBOT_HARNESS_IDLE_SECONDS`、`ROBOT_HARNESS_MAX_WARM` 可配。
- 访客对话按浏览器会话关联 Harness 持久化会话；选择或识别成员后改为关联该成员的独立会话，换浏览器选择同一成员也可接续。匿名历史不会自动归给某个人；首次接入的访客仍会带入旧后端保存的摘要与近期对话。会话保存在本机 `data/harness/`，映射与提醒草稿保存在 SQLite。超长对话的压缩与召回仍待专项验收，不承诺无限记忆。
- “新对话”重置当前成员（或访客）的会话关联、取消其待确认草稿，身份和已保存的提醒不受影响；界面会说明成员记忆被重置。历史 Harness 文件不会物理删除。当前为本机家庭共享设备，没有账号鉴权，手动选择成员不是身份认证。
- 可配置 `ROBOT_HARNESS_COMMAND`、`ROBOT_HARNESS_TIMEOUT_SECONDS`、`ROBOT_HARNESS_MAX_TOKENS`。需要临时回退时，停掉当前服务后执行 `ROBOT_AGENT_BACKEND=legacy ./run.sh`；旧后端仍使用原有 OpenAI 兼容配置。
- 中文朗读默认使用百炼云端 `qwen3-tts-flash` 的 Cherry 女声（实测首片约 0.37 秒、合成快于实时 3 倍），失败自动回退本地神经语音→系统中文音色。设置页可切换本地 Kokoro `zf_001` 或 Argmax Qwen3-TTS 0.6B 的 Serena、Vivian、Dylan、Eric、Uncle Fu 中文声线；选择保存在本机 `data/voice-preference.json`，无需重启。Argmax 1.7B 当前测试生成不稳定，因此未开放。Argmax 的 CLI 和 0.6B 模型路径可用 `ROBOT_ARGMAX_CLI`、`ROBOT_ARGMAX_MODEL_DIR` 覆盖。多音字、姓名和中英混读仍需试听验收。
- `ROBOT_TTS_BACKEND=auto` 按所选音色走，失败逐级降级；`kokoro` 为严格模式，`system` 为系统模式。健康接口会显示当前后端和降级原因。
- 麦克风识别默认走服务端 `POST /api/stt`（百炼 `qwen3-asr-flash`，实测整句约 0.5 秒）：页面用 AudioWorklet 录音（16kHz），松手后整段上传识别，文字出现在输入框、可修改后发送。全静音录音在服务端直接拦截，不送模型（LLM 架构的 ASR 对静音会幻觉出文本）。浏览器不支持录音接口或服务端识别关闭（`ROBOT_STT_BACKEND=browser`）时，自动回退 Web Speech API 实时识别。麦克风和摄像头需用户在浏览器授权。

语音模型来源：[kokoro-onnx 官方发布](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1)、[Kokoro 中文模型卡](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)。模型卡标注 Apache-2.0；分发时同时核对各组件和音色的许可。下载脚本校验 SHA-256，macOS CPU 使用完整模型以避开 int8 ConvInteger 不兼容。


## 当前说话人与个人记忆

页面顶部显示当前成员。进入“家庭”，打开摄像头并点击“识别并开始聊天”：只有单个已知人脸匹配成功才绑定成员；多人、未知或未检测到人脸时切换到新的访客会话，提示手动选择。也可以在顶部选择已注册成员并点击“开始聊天”。注册成功后立即以该成员身份聊天。同名重新注册更新特征、保留成员编号及记忆。

成员姓名和编号进入 Agent 上下文；个人历史按成员独立保存，再次识别/选择同一人会接续。换人时清空屏幕上的上一位对话、停止朗读，加载新成员提醒；正在回答或录音时需先结束再切换。多个窗口共享一个浏览器身份时会定期同步，旧页面提交会校验身份版本。

“提醒我”默认归当前成员，访客默认归家庭共享；“提醒全家”可以明确创建共享提醒。对话工具与新版页面展示当前成员个人提醒及家庭共享提醒，不展示其他成员个人事项；换人后旧草稿不能直接确认。到点播放使用原提醒系统，并称呼所属成员。旧提醒中没有成员归属的记录继续视为家庭共享，不自动猜归属。无 `session_id` 的旧提醒接口保留家庭列表兼容行为，这些本机接口不作为权限隔离或账号认证。

当前不会持续监测换人，也没有声纹识别；脸出现在镜头里不等于确定谁正在发声，因此页面持续显示身份，换人请重新识别或选择。实际摄像头识别准确率仍需在设备上验收。

照片只发送给本机机器人 HTTP 服务。注册/识别由本地 dlib 或 OpenCV 处理，临时原图处理后删除；注册特征保存在本机 SQLite 的 `people.face_encoding`，备份也可能包含这些特征。Harness / 百炼不接收照片、人脸特征或人脸模型，只接收当前成员姓名、编号、对话内容及本次工具结果；识别本身不调用云模型。启用云端语音时，百炼额外接收录音音频（识别）和回答文本（合成），见「本地运行」的隐私边界。

## 提醒边界

支持“一分钟后提醒我削笔”，也支持先说“两分钟后提醒我”、再补充事项。Agent 追问缺少的信息，调用工具生成草稿；点击确认或在当前对话明确说“确认”后才保存到原提醒系统。相对时间按用户提出时刻计算；修改后需要确认最新版本，重复点击不会创建重复提醒。口头生成的文本不是保存成功的依据。

时间查询读取本机时钟；天气查询 Open-Meteo，需要城市信息和网络。天气失败会反馈错误。当前工具范围不包含音乐、家电或通用电脑操作。

提醒按真实时间比较，播放成功后才标为送达；失败保留待提醒并每 15 秒重试。当前只支持一次性提醒；服务关闭时不会提醒，重新启动后会尝试播放已到期提醒。请保持一个服务进程，避免重复播放。

## 接口与验证

保留原有健康检查、提醒、人脸注册/识别、语音识别、聊天和命令接口。新增：

- `POST /api/command/stream?text=...`：SSE，start/status/result/delta/error/done；Harness 的 delta 当前承载完整回答。
- `POST /api/voice/audio`：JSON `{"text":"你好"}`，返回中文 WAV，不在服务器扬声器自动播放。
- `POST /api/stt`：multipart `audio`（16bit 单声道 WAV），返回 `{"text","silent","stt_ms","stats"}`；全静音直接返回 `silent:true`，不调模型。
- `GET /static/worklet.js`：麦克风 AudioWorklet（采集、重采样 16k、电平上报）。
- `GET /api/health`：当前 Agent 后端、模型最近调用状态、实际语音后端与降级原因、识别后端（`stt` 字段）。
- `GET /api/conversations/{session_id}/drafts`：恢复待确认提醒。
- `POST /api/reminder-drafts/{draft_id}/confirm` 与 `PATCH /api/reminder-drafts/{draft_id}`：校验会话与草稿版本后确认或编辑。

```bash
.venv-voice/bin/python -B -m unittest discover -s tests -v
```

单元测试使用临时数据库，不修改家庭成员和提醒数据。可选的真实 Agent 联调会读取本机密钥并产生模型请求，提醒只写入临时数据库，不启动播放：

```bash
.venv-voice/bin/python -B scripts/harness_smoke.py
.venv-voice/bin/python -B scripts/identity_smoke.py
```

已验证真实模型的时间、天气、多轮补全、提醒修改/确认/取消、旧摘要迁移与会话隔离，以及服务端语音识别（整句约 0.5 秒、逐字正确）和百炼云端音色合成（首片约 0.37 秒）。浏览器麦克风录音链路、人脸、到点声音播放仍需设备实测。

第二阶段再接桌面机器人外壳与交互硬件，之后才考虑移动、自主感知与行动。

## 产品设计预览

当前运行页面已按产品设计收敛为「陪伴 / 提醒 / 家庭 / 设置」四个入口：主屏展示机器人形象、对话和下一条提醒；摄像头只在进入家庭页并点击后打开；提醒支持创建、确认、修改和取消。设计说明和静态可点击原型位于 [`docs/产品设计-v1/`](docs/产品设计-v1/)。静态原型不会连接摄像头、模型或数据库，只用于评审信息结构。
