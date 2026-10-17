# 家庭机器人

目标：识别已录入的家庭成员，听懂说话，会说话，并按时提醒事情。

## 本地运行

```bash
cd /Users/yuanlw/robot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

打开 http://127.0.0.1:8000 。第一版先验证提醒接口，语音和人脸能力通过可选依赖接入。

## 当前接口

- `GET /api/health`：健康检查
- `GET /api/reminders`：查看活动提醒
- `POST /api/reminders`：创建提醒，JSON 为 `{"text":"练琴","due_at":"2026-10-18T20:00:00+08:00"}`
- `DELETE /api/reminders/{id}`：取消提醒
- `POST /api/faces/enroll`：上传一张只含一个人的照片注册
- `POST /api/faces/recognize`：上传照片识别已注册成员
- `POST /api/speak?text=你好`：调用本机 TTS（未安装时返回 text-only）

## 设计原则

核心业务只依赖抽象能力，设备驱动和模型放在适配层，先在笔记本开发，之后迁移到 Mac mini 和树莓派。低置信度的人脸识别必须返回“未知”，提醒的删除和修改需要确认。
