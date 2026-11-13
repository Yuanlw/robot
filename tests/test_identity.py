import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import db, identity, agent_store, reminders
from app.main import app, faces
from app.face import FaceService
from app.agent_tools import execute
from app.harness import agent, HarnessService
from app.config import settings


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for obj, attr, value in [(db, 'DB_PATH', Path(self.tmp.name)/'robot.db'), (settings,'data_dir',Path(self.tmp.name))]:
            p=patch.object(obj,attr,value);p.start();self.addCleanup(p.stop)
        db.init_db();agent_store.init_agent_db()
        self.a=FaceService()._save('小林',b'private-face-feature-A')
        self.b=FaceService()._save('小周',b'private-face-feature-B')
        self.client=TestClient(app, headers={'X-Robot-Client': 'test'})
        self.now=datetime.now().astimezone()

    def select(self, member, browser='browser'):
        state=identity.public(browser)
        r=self.client.put(f'/api/conversations/{browser}/identity',json={'person_id':member,'revision':state['revision']})
        self.assertEqual(r.status_code,200,r.text)
        return r.json()

    def draft(self, browser='browser'):
        member=identity.current(browser)['person']
        sid=agent_store.session(browser)['harness_id']
        context={'session_id':sid,'person_id':member['id'] if member else None,'received_at':self.now.isoformat()}
        row=execute('prepare_reminder',{'text':'削笔','due_at':(self.now+timedelta(minutes=10)).isoformat()},context)
        return sid,row,context

    def test_member_context_survives_switch_and_another_browser(self):
        guest=agent_store.session('browser')['harness_id']
        self.select(self.a['id']);a=agent_store.session('browser')['harness_id'];agent_store.mark_started('browser')
        self.select(self.b['id']);b=agent_store.session('browser')['harness_id']
        self.assertEqual(len({guest,a,b}),3)
        self.select(self.a['id'])
        self.assertEqual(agent_store.session('browser')['harness_id'],a)
        self.assertEqual(agent_store.session('browser')['started'],1)
        self.select(self.a['id'],'another-browser')
        self.assertEqual(agent_store.session('another-browser')['harness_id'],a)
        self.assertIs(agent.lock('browser'),agent.lock('another-browser'))
        self.select(None)
        self.assertNotIn(agent_store.session('browser')['harness_id'],{guest,a,b})

    def test_recognition_binds_only_single_known_person(self):
        for found,expected in [([{'person_id':self.a['id'],'name':'小林'}],self.a['id']),
                                ([{'person_id':self.a['id'],'name':'小林'},{'person_id':self.b['id'],'name':'小周'}],None),
                                ([{'person_id':None,'name':'未知'}],None),([],None)]:
            before=identity.public('browser')
            with patch.object(faces,'recognize',return_value={'faces':found,'count':len(found)}):
                r=self.client.post('/api/faces/recognize',data={'session_id':'browser','identity_revision':before['revision']},files={'image':('face.jpg',b'test-image','image/jpeg')})
            self.assertEqual(r.status_code,200,r.text)
            actual=r.json()['identity']['person']
            self.assertEqual(actual['id'] if actual else None,expected)

    def test_stale_identity_and_inflight_switch_rejected(self):
        old=identity.public('browser')['revision']
        self.select(self.a['id'])
        r=self.client.put('/api/conversations/browser/identity',json={'person_id':self.b['id'],'revision':old})
        self.assertEqual(r.status_code,409)
        async def locked():
            async with agent.lock('browser'):
                r=self.client.put('/api/conversations/browser/identity',json={'person_id':self.b['id'],'revision':identity.public('browser')['revision']})
                self.assertEqual(r.status_code,409)
        asyncio.run(locked())

    def test_personal_reminder_and_cross_member_protection(self):
        self.select(self.a['id']);sid,draft,context=self.draft()
        row=agent_store.confirm(sid,draft['id'],draft['revision'])
        self.assertEqual(row['person_id'],self.a['id'])
        self.assertEqual(reminders.due_now(self.now+timedelta(hours=1))[0]['person_name'],'小林')
        self.select(self.b['id'])
        self.assertEqual(self.client.get('/api/reminders',params={'session_id':'browser'}).json(),[])
        self.assertEqual(self.client.delete('/api/reminders/'+str(row['id']),params={'session_id':'browser'}).status_code,404)
        other=agent_store.session('browser')['harness_id']
        for name,args in [('cancel_reminder',{'reminder_id':row['id']}),('prepare_reminder',{'reminder_id':row['id'],'text':'喝水'})]:
            with self.assertRaises(ValueError):execute(name,args,{**context,'session_id':other,'person_id':self.b['id']})
        self.assertEqual(reminders.get(row['id'])['status'],'active')
        self.assertEqual(self.client.post('/api/reminder-drafts/'+draft['id']+'/confirm',json={'session_id':'browser','revision':1}).status_code,409)

    def test_shared_and_form_reminder_ownership(self):
        state=self.select(self.a['id']);due=(self.now+timedelta(hours=1)).isoformat()
        params={'session_id':'browser','identity_revision':state['revision']}
        personal=self.client.post('/api/reminders',params=params,json={'text':'喝水','due_at':due}).json()
        shared=self.client.post('/api/reminders',params=params,json={'text':'全家开饭','due_at':due,'scope':'household'}).json()
        self.assertEqual(personal['person_id'],self.a['id']);self.assertIsNone(shared['person_id'])
        self.select(self.b['id'])
        visible=self.client.get('/api/reminders',params={'session_id':'browser'}).json()
        self.assertEqual([r['id'] for r in visible],[shared['id']])
        r=self.client.post('/api/reminders',params=params,json={'text':'过期页面','due_at':due})
        self.assertEqual(r.status_code,409)

    def test_photo_temporary_file_removed_on_error_and_success(self):
        paths=[]
        def recognition(path):
            paths.append(path)
            self.assertEqual(Path(path).read_bytes(),b'local-photo')
            return {'faces':[], 'count':0}
        for fail in (False,True):
            with patch.object(faces,'recognize',side_effect=recognition if not fail else lambda path:(paths.append(path), (_ for _ in ()).throw(ValueError('bad image')))[1]):
                r=self.client.post('/api/faces/recognize',files={'image':('../../unsafe.jpg',b'local-photo','image/jpeg')})
                self.assertEqual(r.status_code,400 if fail else 200)
            self.assertFalse(Path(paths[-1]).exists())
            self.assertTrue(Path(paths[-1]).name.startswith('robot-face-'))

    def test_recognizer_returns_stable_id_and_unknown_has_no_id(self):
        from types import SimpleNamespace
        for member,value in [(self.a,0.1),(self.b,0.8)]:
            FaceService()._save(member['name'],json.dumps({'backend':'dlib','values':[value]}).encode())
        encoding=[0.1]
        backend=SimpleNamespace(load_image_file=lambda path:None,face_locations=lambda image:[0],
            face_encodings=lambda image,locations:[encoding],face_distance=lambda known,value:[abs(known[0][0]-value[0])])
        with patch.object(faces,'_dlib',return_value=backend):
            self.assertEqual(faces.recognize('local')['faces'][0]['person_id'],self.a['id'])
            encoding[0]=2.0
            self.assertIsNone(faces.recognize('local')['faces'][0]['person_id'])

    def test_reenrollment_stable_id_and_no_features_in_people_api(self):
        saved=FaceService()._save('小林',b'new-private-face-feature')
        self.assertEqual(saved['id'],self.a['id'])
        r=self.client.get('/api/people')
        people=r.json()
        self.assertEqual([{k:p[k] for k in ('id','name')} for p in people],[self.a,self.b])
        self.assertTrue(all('samples' in p for p in people))  # 2026-11-13: 新增样本数字段
        self.assertNotIn('feature',r.text)

    def test_clear_only_current_member_and_legacy_guest_migration(self):
        with db.connection() as c:c.execute('INSERT INTO agent_sessions VALUES (?,?,?)',('legacy-browser','old-session',1))
        self.assertEqual(agent_store.session('legacy-browser')['harness_id'],'old-session')
        self.select(self.a['id']);a=agent_store.session('browser')['harness_id']
        self.select(self.b['id']);b=agent_store.session('browser')['harness_id']
        self.client.delete('/api/conversations/browser')
        self.assertEqual(identity.public('browser')['person']['id'],self.b['id'])
        self.assertNotEqual(agent_store.session('browser')['harness_id'],b)
        self.select(self.a['id']);self.assertEqual(agent_store.session('browser')['harness_id'],a)

    def test_model_receives_name_but_no_photo_or_other_member_data(self):
        self.select(self.a['id'])
        reminders.create('小周秘密事项',self.now+timedelta(hours=1),self.b['id'])
        captured=[]
        class FakeSDK:
            def __init__(self,path,home,event):
                self.answers=['你好，小林'];captured.append(json.loads(path.read_text()))
            async def start(self):pass
            async def rpc(self,method,params):
                if method=='session/prompt':captured.append(params)
            async def close(self):pass
        with patch('app.harness.SDK',FakeSDK),patch.object(settings,'llm_api_key','test-only'),patch.object(settings,'harness_command','/usr/bin/true'):
            asyncio.run(HarnessService().run('browser','你认识我吗'))
        text=json.dumps(captured,ensure_ascii=False)
        self.assertIn('小林',text)
        for forbidden in ['private-face','face_encoding','小周秘密事项','base64']:
            self.assertNotIn(forbidden,text)

    def test_delete_person_full_cleanup(self):
        a_id=self.a['id']
        browser='browser-delete'
        self.select(a_id,browser)
        revision_before=identity.public(browser)['revision']
        sid=agent_store.session(browser)['harness_id']
        _,draft_row,_=self.draft(browser)
        delivered=reminders.create('喝水',self.now+timedelta(hours=1),person_id=a_id)
        reminders.mark_delivered(delivered['id'])
        r=self.client.delete(f'/api/people/{a_id}')
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['deleted'],a_id)
        self.assertIsNone(identity.person(a_id))
        # 已送达提醒保留为历史；人脸特征随行删除；会话映射清理
        with db.connection() as c:
            self.assertIsNone(c.execute('SELECT face_encoding FROM people WHERE id=?',(a_id,)).fetchone())
            self.assertEqual(c.execute('SELECT COUNT(*) FROM reminders WHERE id=?',(delivered['id'],)).fetchone()[0],1)
            self.assertIsNone(c.execute('SELECT 1 FROM agent_sessions WHERE browser_id=?',('person:'+str(a_id),)).fetchone())
        self.assertEqual(agent_store.drafts(sid),[])
        # 绑定该成员的浏览器退回访客，revision 递增触发页面自动换屏
        state=identity.current(browser)
        self.assertIsNone(state['person'])
        self.assertEqual(state['source'],'guest')
        self.assertGreater(state['revision'],revision_before)

    def test_delete_person_blocked_by_active_personal_reminder(self):
        a_id=self.a['id']
        browser='browser-delete-busy'
        self.select(a_id,browser)
        reminders.create('吃药',self.now+timedelta(hours=2),person_id=a_id)
        r=self.client.delete(f'/api/people/{a_id}')
        self.assertEqual(r.status_code,409)
        self.assertIn('提醒',r.json()['detail'])
        self.assertIsNotNone(identity.person(a_id))

    def test_delete_person_404(self):
        self.assertEqual(self.client.delete('/api/people/999999').status_code,404)

    def test_delete_person_while_conversation_locked(self):
        a_id=self.a['id']
        scope='person:'+str(a_id)
        agent.locks[scope]=asyncio.Lock()
        self.addCleanup(agent.locks.pop,scope,None)
        async def hold():
            async with agent.locks[scope]:
                await asyncio.sleep(0.2)
        async def scenario():
            task=asyncio.create_task(hold())
            await asyncio.sleep(0.05)
            return self.client.delete(f'/api/people/{a_id}')
        r=asyncio.run(scenario())
        self.assertEqual(r.status_code,409)
        self.assertIn('对话',r.json()['detail'])
        self.assertIsNotNone(identity.person(a_id))


if __name__=='__main__':unittest.main()
