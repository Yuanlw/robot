"""MCP stdio tools backed by the robot's actual database and reminder scheduler."""
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from . import agent_store, reminders, identity, study
from .config import settings
from .db import connection, init_db


def schema(name, description, properties=None, required=None):
    return {'name': name, 'description': description, 'inputSchema': {
        'type': 'object', 'properties': properties or {}, 'required': required or [], 'additionalProperties': False}}


TOOLS = [
    schema('get_current_time', '获取准确时间。相对提醒优先使用提出时间那一轮应用提供的 input_reference_id；补充事项时沿用该基准。'),
    schema('get_weather', '查询指定城市的当前天气和今天/明天/后天天气预报。地点不明确先追问，查询失败不能编造。', {
        'city': {'type': 'string'}, 'day': {'type': 'integer', 'enum': [0, 1, 2], 'description': '0今天，1明天，2后天，默认0'}}, ['city']),
    schema('prepare_reminder', '创建待确认草稿，尚未生效。缺少时间或具体事项先追问。修改草稿传 draft_id；修改已生效提醒先查询并传 reminder_id。仅改内容时不传时间，保留原时间。', {
        'scope': {'type': 'string', 'enum': ['personal', 'household'], 'description': '默认当前成员个人提醒；用户明确说全家/大家才用household。访客默认家庭共享，必须告知归属；若要求个人提醒，先请用户选择成员。'},
        'text': {'type': 'string'}, 'due_at': {'type': 'string', 'description': '明确钟点使用带时区 ISO8601'},
        'delay_seconds': {'type': 'integer', 'description': '相对时长，1分钟=60，不传 due_at'},
        'reference_id': {'type': 'string'}, 'draft_id': {'type': 'string'}, 'reminder_id': {'type': 'integer'}}),
    schema('confirm_reminder', '仅用户明确确认当前草稿后调用。应用验证本轮确认授权，拒绝模型自行确认。', {
        'id': {'type': 'string'}, 'revision': {'type': 'integer'}}, ['id', 'revision']),
    schema('list_reminders', '查询当前家庭待提醒事项和本会话待确认草稿，修改或取消已有提醒前先查询。'),
    schema('cancel_reminder', '取消用户明确指定的提醒或草稿。传 reminder_id 或 draft_id，二选一。', {
        'reminder_id': {'type': 'integer'}, 'draft_id': {'type': 'string'}}),
    schema('study_extend', '延长当前学习或休息阶段。当用户说"再给我N分钟"时调用。', {
        'minutes': {'type': 'integer', 'description': '延长分钟数，默认5'}}, []),
    schema('study_pause', '暂停学习进入休息。'),
    schema('study_resume', '从休息恢复学习。'),
    schema('study_end', '结束本次学习会话，播报总时长。'),
    schema('study_review', '查看学习报告，回答"最近学得怎么样"类问题。', {
        'period': {'type': 'string', 'enum': ['today', 'week', 'month'], 'description': '默认week'}}, []),
    schema('study_suggest', '根据历史数据给出学习节奏建议（最佳时长、高效时段）。'),
    schema('study_rate', '用户自评本次学习质量时调用，如"今天学得不错"。', {
        'score': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'description': '10为最佳'},
        'tags': {'type': 'array', 'items': {'type': 'string'}, 'description': '场景标签，如["晚上","有点吵"]'}}, ['score']),
]


def fetch_json(url):
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def weather(city, day=0):
    if not isinstance(city, str) or not city.strip() or len(city) > 80 or day not in (0, 1, 2):
        raise ValueError('请提供城市名称及今天、明天或后天')
    known = {'北京': (39.9042, 116.4074), '上海': (31.2304, 121.4737),
             '广州': (23.1291, 113.2644), '深圳': (22.5431, 114.0579),
             '杭州': (30.2741, 120.1551), '成都': (30.5728, 104.0668)}
    name = city.strip().removesuffix('市')
    if name in known:
        lat, lon = known[name]
        location = name
    else:
        matches = fetch_json('https://geocoding-api.open-meteo.com/v1/search?' + urlencode({
            'name': city, 'count': 5, 'language': 'zh', 'format': 'json'})).get('results', [])
        if not matches:
            raise ValueError('未查到该城市，请提供省份和城市名称，或尝试城市英文名')
        if len(matches) > 1:
            return {'needs_clarification': True, 'locations': [
                {'name': r['name'], 'region': r.get('admin1'), 'country': r.get('country')} for r in matches]}
        item = matches[0]
        lat, lon = item['latitude'], item['longitude']
        location = ' '.join(filter(None, [item.get('country'), item.get('admin1'), item['name']]))
    url = 'https://api.open-meteo.com/v1/forecast?' + urlencode({
        'latitude': lat, 'longitude': lon, 'timezone': 'auto', 'forecast_days': 3,
        'current': 'temperature_2m,relative_humidity_2m,precipitation,weather_code',
        'daily': 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max'})
    data = fetch_json(url)
    return {'location': location, 'source': 'Open-Meteo', 'retrieved_at': datetime.now().astimezone().isoformat(),
            'timezone': data['timezone'], 'current': data['current'] if day == 0 else None,
            'current_units': data['current_units'],
            'forecast': {k: v[day] for k, v in data['daily'].items()}, 'forecast_units': data['daily_units']}


def execute(name, args, context):
    sid = context['session_id']
    if name == 'get_current_time':
        rid = uuid.uuid4().hex
        with connection() as c:
            c.execute('INSERT INTO agent_clocks VALUES (?,?,?)', (rid, sid, context['received_at']))
        return {'now': datetime.now(ZoneInfo(settings.timezone)).isoformat(), 'timezone': settings.timezone,
                'reference_id': rid, 'input_received_at': context['received_at']}
    if name == 'get_weather':
        return weather(args['city'], args.get('day', 0))
    if name == 'prepare_reminder':
        return agent_store.prepare(sid, args, context['received_at'], context.get('person_id'))
    if name == 'confirm_reminder':
        if context.get('confirmations', {}).get(args['id']) != args['revision']:
            raise ValueError('用户尚未明确确认此版本草稿，请让用户点击确认提醒，或回答“确认”')
        return agent_store.confirm(sid, args['id'], args['revision'])
    if name == 'list_reminders':
        return {'active': identity.visible_reminders(context.get('person_id')), 'drafts': agent_store.drafts(sid)}
    if name == 'cancel_reminder':
        if bool(args.get('draft_id')) == bool(args.get('reminder_id')):
            raise ValueError('只能选择一条草稿或提醒，不能同时取消两者')
        if args.get('draft_id'):
            return agent_store.cancel_draft(sid, args['draft_id'])
        identity.require_reminder(args['reminder_id'], context.get('person_id'))
        if not reminders.cancel(args['reminder_id']):
            raise ValueError('提醒不存在或已经送达/取消')
        return {'cancelled': True, 'reminder_id': args['reminder_id']}
    
    # 学习陪伴工具
    if not settings.study_enabled:
        raise ValueError('学习陪伴功能未启用')
    person_id = context.get('person_id')
    if not person_id:
        raise ValueError('需要先选择身份')
    
    if name == 'study_extend':
        session = study.get_current_session(person_id)
        if not session:
            raise ValueError('当前没有进行中的学习会话')
        minutes = args.get('minutes', 5)
        study.record_reminder(session['id'], 'extend', str(minutes))
        return {'ok': True, 'message': f'好的，再给你 {minutes} 分钟'}
    
    if name == 'study_pause':
        session = study.get_current_session(person_id)
        if not session:
            raise ValueError('当前没有进行中的学习会话')
        study.pause_session(session['id'])
        return {'ok': True, 'message': '已暂停，好好休息'}
    
    if name == 'study_resume':
        session = study.get_current_session(person_id)
        if not session:
            raise ValueError('当前没有进行中的学习会话')
        study.resume_session(session['id'])
        return {'ok': True, 'message': '继续加油！'}
    
    if name == 'study_end':
        session = study.get_current_session(person_id)
        if not session:
            raise ValueError('当前没有进行中的学习会话')
        elapsed = study.compute_elapsed(session)
        study.end_session(session['id'])
        study_min = elapsed['study_seconds'] // 60
        break_min = elapsed['break_seconds'] // 60
        summary = f"今天学了 {study_min} 分钟"
        if break_min > 0:
            summary += f"，休息了 {break_min} 分钟"
        summary += "，辛苦了！"
        return {'ok': True, 'message': summary, 'elapsed': elapsed}

    if name == 'study_review':
        from . import study_report
        period = args.get('period', 'week')
        person = identity.person(person_id)
        report = study_report.build_report(person_id, person['name'] if person else '', period)
        # 报告给模型时只给标题与 markdown，stats/profile 字段太长会挤占上下文
        return {'headline': report['headline'], 'markdown': report['markdown']}

    if name == 'study_suggest':
        from . import study_profiler
        return study_profiler.suggest(person_id)

    if name == 'study_rate':
        session = study.get_current_session(person_id)
        if not session:
            raise ValueError('当前没有进行中的学习会话')
        study.rate_session(session['id'], args['score'], args.get('tags') or [])
        return {'ok': True, 'message': '记下来了，谢谢自评！'}

    raise ValueError('未知工具')


def main():
    init_db()
    agent_store.init_agent_db()
    study.init_study_db()
    from . import vision as _vision
    _vision.init_vision_db()
    for line in sys.stdin:
        request = json.loads(line)
        if 'id' not in request:
            continue
        method, params = request['method'], request.get('params', {})
        if method == 'initialize':
            result = {'protocolVersion': params['protocolVersion'], 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'family-robot-tools', 'version': '1.0'}}
        elif method == 'tools/list':
            result = {'tools': TOOLS}
        elif method == 'tools/call':
            try:
                context = json.loads(Path(os.environ['ROBOT_AGENT_CONTEXT']).read_text())
                value = execute(params['name'], params.get('arguments', {}), context)
                result = {'content': [{'type': 'text', 'text': json.dumps(value, ensure_ascii=False)}]}
            except Exception as exc:
                # Do not expose request headers, credentials or stack traces to the model.
                message = str(exc) if isinstance(exc, (ValueError, KeyError)) else '工具暂时不可用，请稍后再试'
                result = {'isError': True, 'content': [{'type': 'text', 'text': message}]}
        elif method == 'ping':
            result = {}
        else:
            print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32601, 'message': 'Unknown method'}}), flush=True)
            continue
        print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
