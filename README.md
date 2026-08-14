# AI Meeting Transcription

面向 Linux 私有化部署的网页版 AI 会议工具。页面只保留真实可用功能：

- 创建会议后点击开始会议，浏览器麦克风以 16k mono Float32 PCM 推给 Qwen ASR streaming，实时文字直接输出。
- 上传音频文件，调用 Qwen ASR 文件转写接口返回文字。
- 基于当前会议文字调用本地 vLLM 生成 AI 纪要总结。

项目不会预置示例会议或模拟转写内容。

## 服务组成

需要同时运行三个核心服务：

```text
8006  Qwen3-ASR-0.6B 统一 ASR 服务，用于实时会议转写和上传音频转写
8012  Qwen3.6-35B-A3B vLLM 服务，用于 AI 纪要
8011  meeting-ai 后端 API
5173  meeting-ai 前端开发服务
```

已有 LLM 启动脚本在 `/home/regchen/Chuyi/model_run`：

```bash
/home/regchen/Chuyi/model_run/run_qwen36_B35_A3B_vllm.sh
```

统一 ASR 启动脚本：

```bash
cd /home/regchen/Chuyi/meeting-ai-mvp/backend
./start_streaming_asr.sh
```

## 后端配置

`backend/.env`：

```bash
LLM_BASE_URL=http://127.0.0.1:8012/v1
LLM_MODEL=Qwen3.6-35B-A3B-NVFP4
LLM_API_KEY=EMPTY

ASR_BASE_URL=http://127.0.0.1:8006
ASR_MODEL=/home/regchen/Chuyi/models/Qwen3-ASR-0.6B
ASR_API_KEY=EMPTY
ASR_TRANSCRIBE_PATH=/v1/audio/transcriptions
ASR_MAX_RETRIES=2

ASR_STREAM_BASE_URL=http://127.0.0.1:8006
ASR_STREAM_PORT=8006
ASR_STREAM_MODEL=/home/regchen/Chuyi/models/Qwen3-ASR-0.6B
ASR_STREAM_GPU_MEMORY_UTILIZATION=0.12
ASR_STREAM_MAX_MODEL_LEN=4096
ASR_STREAM_CHUNK_SIZE_SEC=1.0
ASR_STREAM_UNFIXED_CHUNK_NUM=4
ASR_STREAM_UNFIXED_TOKEN_NUM=5
```

如果未配置对应服务地址，接口会返回错误，不会生成 mock 文本。

## 开发启动

启动统一 ASR：

```bash
cd /home/regchen/Chuyi/meeting-ai-mvp/backend
./start_streaming_asr.sh
```

启动 vLLM 纪要模型：

```bash
/home/regchen/Chuyi/model_run/run_qwen36_B35_A3B_vllm.sh
```

启动应用后端，使用 `meeting-ai` conda 环境：

```bash
cd /home/regchen/Chuyi/meeting-ai-mvp/backend
conda activate meeting-ai
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8011
```

启动前端：

```bash
cd /home/regchen/Chuyi/meeting-ai-mvp/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

访问：

```text
http://服务器IP:5173
```

## 内网麦克风访问

浏览器默认只允许 HTTPS、`localhost`、`127.0.0.1` 页面调用麦克风。公司内网用户直接访问 `http://服务器IP:5173` 时，上传音频可用，但实时录音通常会被浏览器拦截。

推荐给内网地址配置 HTTPS。开发临时方案可以让公司 IT 对 Chrome/Edge 下发企业策略，把当前地址加入允许列表：

```json
{
  "UnsafelyTreatInsecureOriginAsSecure": ["http://服务器IP:5173"]
}
```

策略下发后，用户需要重启浏览器并重新打开页面。

## 当前接口

```text
GET    /api/health
GET    /api/meetings
POST   /api/meetings
GET    /api/meetings/{id}
POST   /api/meetings/{id}/audio
POST   /api/meetings/{id}/stream/start
POST   /api/meetings/{id}/stream/chunk
POST   /api/meetings/{id}/stream/finish
POST   /api/meetings/{id}/summary
```
