"""Opt-in real Harness test, temporary members/DB, no camera or speaker playback."""
import asyncio,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
from dotenv import dotenv_values
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

async def main():
    cfg=dotenv_values(os.environ.get('ROBOT_SMOKE_CONFIG',str(ROOT/'.env')))
    key=os.environ.get('ROBOT_LLM_API_KEY') or cfg.get('ROBOT_LLM_API_KEY')
    if not key:raise RuntimeError('本地未配置模型密钥')
    with tempfile.TemporaryDirectory(prefix='robot-identity-smoke-') as tmp:
        os.environ.update(ROBOT_DATA_DIR=tmp,ROBOT_LLM_API_KEY=key,ROBOT_AGENT_BACKEND='harness')
        import httpx
        from app.main import app,faces
        from app.db import init_db
        from app import agent_store,identity
        from app.face import FaceService
        init_db();agent_store.init_agent_db()
        a=FaceService()._save('小林',b'local-test-feature-a');b=FaceService()._save('小周',b'local-test-feature-b')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',timeout=90) as client:
            async def send(text,browser='live'):
                state=identity.public(browser)
                r=await client.post('/api/command/stream',params={'text':text,'session_id':browser,'identity_revision':state['revision']})
                events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: ')]
                assert events[-1]['type']=='done',events
                answer=''.join(x['text'] for x in events if x['type']=='delta')
                print(json.dumps({'speaker':state['person'],'prompt':text,'reply':answer,'events':events},ensure_ascii=False),flush=True)
                return answer
            async def select(member,browser='live'):
                state=identity.public(browser)
                r=await client.put('/api/conversations/'+browser+'/identity',json={'person_id':member,'revision':state['revision']})
                assert r.status_code==200,r.text
            with patch.object(faces,'recognize',return_value={'faces':[{'person_id':a['id'],'name':a['name']}],'count':1}):
                r=await client.post('/api/faces/recognize',data={'session_id':'live','identity_revision':0},files={'image':('test.jpg',b'local-only-test-photo','image/jpeg')})
                assert r.status_code==200,r.text
            assert '小林' in await send('你知道我是谁吗？')
            await send('我最喜欢的饮料是普洱茶，请记住。')
            await select(b['id'])
            answer=await send('我是谁，你知道我最喜欢什么饮料吗？')
            assert '小周' in answer and '普洱' not in answer,answer
            await select(a['id'],'another-browser')
            assert '普洱' in await send('你记得我最喜欢什么饮料吗？','another-browser')
            await send('十分钟后提醒我喝茶','another-browser')
            sid=agent_store.session('another-browser')['harness_id'];draft=agent_store.drafts(sid)[0]
            assert draft['person_id']==a['id'],draft
            await send('确认','another-browser')
            visible=(await client.get('/api/reminders',params={'session_id':'another-browser'})).json()
            assert len(visible)==1 and visible[0]['person_name']=='小林',visible
            assert (await client.get('/api/reminders',params={'session_id':'live'})).json()==[]
            await select(None,'another-browser')
            answer=await send('你知道我是谁，喜欢什么饮料吗？','another-browser')
            assert '普洱' not in answer and '小林' not in answer,answer
            print('IDENTITY_LIVE_CHECKS=PASS',flush=True)

if __name__=='__main__':asyncio.run(main())
