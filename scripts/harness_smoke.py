"""Opt-in live API test: real model/tools, isolated temporary DB, no speaker playback."""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def main():
    config = dotenv_values(os.environ.get('ROBOT_SMOKE_CONFIG', str(ROOT / '.env')))
    key = os.environ.get('ROBOT_LLM_API_KEY') or config.get('ROBOT_LLM_API_KEY')
    if not key:
        raise RuntimeError('请通过环境变量或本地 .env 配置密钥')
    with tempfile.TemporaryDirectory(prefix='robot-harness-live-') as tmp:
        os.environ.update(ROBOT_DATA_DIR=tmp, ROBOT_LLM_API_KEY=key, ROBOT_AGENT_BACKEND='harness')
        import httpx
        from app.main import app
        from app.db import init_db, save_conversation_memory
        from app import agent_store
        init_db(); agent_store.init_agent_db()
        save_conversation_memory('live', '此前对话讨论了画画这项爱好。', [])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test', timeout=90) as client:
            async def send(text, sid='live'):
                response = await client.post('/api/command/stream', params={'text': text, 'session_id': sid})
                assert response.status_code == 200, response.status_code
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
                print(json.dumps({'prompt': text, 'events': events}, ensure_ascii=False), flush=True)
                assert events[-1]['type'] == 'done', events
                return events
            def reply(events):
                return ''.join(e['text'] for e in events if e['type'] == 'delta')
            async def pending():
                return (await client.get('/api/conversations/live/drafts')).json()
            async def active():
                return (await client.get('/api/reminders')).json()
            assert '画画' in reply(await send('之前我们聊过哪项爱好？'))
            assert any(e.get('message') == '正在查看时间' for e in await send('现在北京时间几点？'))
            relative_at = datetime.now().astimezone()
            await send('两分钟后提醒我')
            assert not await pending() and not await active()
            await send('削笔')
            draft = (await pending())[0]
            assert '削笔' in draft['text'] and not await active()
            assert abs((datetime.fromisoformat(draft['due_at']) - relative_at).total_seconds() - 120) < 1
            await send('内容改成喝水，时间不变')
            edited = (await pending())[0]
            assert edited['due_at'] == draft['due_at'] and '喝水' in edited['text']
            await send('改成一分钟后吧')
            await send('确认，就这样')
            assert len(await active()) == 1 and not await pending()
            await send('取消刚才喝水的提醒')
            assert not await active()
            await send('1分钟后提醒我削笔')
            draft = (await pending())[0]
            url = '/api/reminder-drafts/' + draft['id'] + '/confirm'
            data = {'session_id': 'live', 'revision': draft['revision']}
            assert (await client.post(url, json=data)).status_code == 200
            assert (await client.post(url, json=data)).status_code == 200
            assert len(await active()) == 1
            await send('我现在有哪些提醒？')
            assert '画画' in reply(await send('我们之前讨论过什么爱好？'))
            assert '画画' not in reply(await send('我们之前讨论过什么爱好？', sid='isolated'))
            await send('北京现在天气怎么样？')
            assert (await client.delete('/api/conversations/live')).status_code == 200
            assert '画画' not in reply(await send('我们之前讨论过什么爱好？'))
        print('LIVE_API_CHECKS=PASS', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
