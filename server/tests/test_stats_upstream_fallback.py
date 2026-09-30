"""网关侧无用量时，统计页回落到上游真实累计用量。

背景：面板的用量统计（`/api/stats/*`）只累计**经过本管理端网关**（:7864）的调用。
若客户端拿创建上游时那把 api_key 直连 workbuddy2api（:7863）绕过网关，网关一行
都记不到——统计页于是恒为 0，即便上游自己累计了近千次请求。`usage_health` 也检测
不出（它的判据是「今天有请求日志但用量为 0」，而直连场景下请求日志本身也是 0）。

修法：当本地**完全没有任何用量**时，把上游 `/v1/stats` 的真实累计取回来兜底，
让页头卡片与图表反映真实用量；一旦网关记到过流量就立即以本地为准，避免与上游
累计重复计数。

本文件锁住四条性质：
  1. 本地无用量 → summary / daily / hourly / by-model / by-key 全部回落到上游；
  2. 本地有**任何**用量 → 一律不回落到上游（不重复计数）；
  3. realm 过滤按上游模型名前缀（cn: / global:）生效；
  4. 回落只影响展示数字，不改变「今日/本周/总量」的既有字段形态，并在
     usage_health 里如实披露口径（数字来自上游累计，不是网关当日统计）。
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server import config, db  # noqa: E402
from server.routers import stats  # noqa: E402

# 上游 /v1/stats 的真实形态（字段名照抄线上返回）：models[] 带 prompt/completion
# 明细与 credit，模型名带 cn: / global: 前缀。
PAYLOAD = {
    'enabled': True,
    'since': '2026-09-22T04:00:00+08:00',
    'now': '2026-09-30T14:00:00+08:00',
    'uptime_sec': 7200,
    'total': {'model': 'total', 'requests': 1234, 'prompt_tokens': 800000,
              'completion_tokens': 400000, 'credit': 45.6},
    'models': [
        {'model': 'cn:deepseek-v4.1-flash', 'requests': 900,
         'prompt_tokens': 700000, 'completion_tokens': 200000, 'credit': 30.0},
        {'model': 'global:gpt-5.6-sol', 'requests': 334,
         'prompt_tokens': 100000, 'completion_tokens': 200000, 'credit': 15.6},
    ],
}

_USER = {'username': 'a', 'role': 'admin'}


class UpstreamFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_db = config.DB_PATH
        config.DB_PATH = Path(self._tmp.name) / 'f.db'
        db._conn = None
        db.connect()
        db.execute("INSERT INTO api_keys(name,key_hash,prefix,enabled,created_at,quota,used_tokens) "
                   "VALUES('k','h','wbk_test',1,?,0,0)", (int(time.time()),))
        self.kid = db.query_one('SELECT id FROM api_keys LIMIT 1')['id']
        # 兜底有 30 秒模块级缓存，测试之间必须清掉，否则会串味
        stats._upstream_cache.update({'at': 0.0, 'data': None})

    def tearDown(self) -> None:
        try:
            if db._conn is not None:
                db._conn.close()
        except Exception:  # noqa: BLE001
            pass
        db._conn = None
        config.DB_PATH = self._orig_db
        stats._upstream_cache.update({'at': 0.0, 'data': None})
        self._tmp.cleanup()

    # ── 1. 本地无用量 → 全部端点回落 ─────────────────────────────
    def test_summary_falls_back_to_upstream(self) -> None:
        with mock.patch.object(stats, '_fetch_upstream_stats', return_value=PAYLOAD):
            s = stats.summary(user=_USER)
        self.assertEqual(s['today_requests'], 1234)
        self.assertEqual(s['total_requests'], 1234)
        self.assertEqual(s['today_tokens'], 800000 + 400000)
        self.assertIsInstance(s['upstream'], dict)
        self.assertEqual(s['upstream']['requests'], 1234)
        # 必须如实披露口径：数字来自上游累计，不是网关当日统计
        self.assertFalse(s['usage_health']['ok'])
        self.assertIn('上游', s['usage_health']['detail'])
        # 头部模型取上游请求数最多者
        self.assertEqual(s['top_model'], 'cn:deepseek-v4.1-flash')

    def test_by_model_falls_back_and_sorts(self) -> None:
        with mock.patch.object(stats, '_fetch_upstream_stats', return_value=PAYLOAD):
            rows = stats.by_model(user=_USER)
        self.assertEqual([r['name'] for r in rows],
                         ['cn:deepseek-v4.1-flash', 'global:gpt-5.6-sol'])
        self.assertEqual(rows[0]['requests'], 900)

    def test_by_key_falls_back_to_single_labelled_row(self) -> None:
        with mock.patch.object(stats, '_fetch_upstream_stats', return_value=PAYLOAD):
            rows = stats.by_key(user=_USER)
        self.assertEqual(len(rows), 1)
        self.assertIn('直连流量', rows[0]['name'], '必须标注无法按密钥拆分')
        self.assertEqual(rows[0]['requests'], 1234)

    def test_daily_falls_back_to_single_today_point(self) -> None:
        with mock.patch.object(stats, '_fetch_upstream_stats', return_value=PAYLOAD):
            rows = stats.daily(user=_USER)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['day'], time.strftime('%Y-%m-%d'))
        self.assertEqual(rows[0]['requests'], 1234)

    def test_hourly_falls_back_to_current_hour(self) -> None:
        with mock.patch.object(stats, '_fetch_upstream_stats', return_value=PAYLOAD):
            rows = stats.hourly(user=_USER)
        self.assertEqual(len(rows), 24, '小时图固定 24 桶')
        self.assertEqual(sum(r['requests'] for r in rows), 1234, '合计不能翻倍或丢失')
        h = time.localtime().tm_hour
        self.assertEqual(rows[h]['requests'], 1234, '累计放在当前小时这一个桶')

    # ── 2. 本地有任何用量 → 绝不回落（不重复计数） ─────────────────
    def test_no_fallback_when_local_usage_exists(self) -> None:
        db.bump_usage(self.kid, 'global:gpt-5.6-sol', 10, 5, 0.1, realm='global')
        with mock.patch.object(stats, '_fetch_upstream_stats') as m:
            s = stats.summary(user=_USER)
            daily = stats.daily(user=_USER)
            models = stats.by_model(user=_USER)
            keys = stats.by_key(user=_USER)
        m.assert_not_called()
        self.assertIsNone(s['upstream'])
        self.assertEqual(s['today_requests'], 1, '以本地为准，不是上游的 1234')
        self.assertEqual(len(models), 1)
        self.assertNotIn('直连流量', keys[0]['name'])
        self.assertEqual(daily[0]['requests'], 1)

    # ── 3. realm 过滤按上游模型名前缀 ────────────────────────────
    def test_realm_filter_on_upstream_models(self) -> None:
        with mock.patch.object(stats, '_fetch_upstream_stats', return_value=PAYLOAD):
            g = stats.summary(realm='global', user=_USER)
            c = stats.summary(realm='cn', user=_USER)
        self.assertEqual(g['today_requests'], 334)
        self.assertEqual(c['today_requests'], 900)

    def test_realm_filter_skips_other_realm_local_usage(self) -> None:
        """本地只有 cn 用量时，看 global 仍应回落（`_has_local_usage` 按版本判断）。"""
        db.bump_usage(self.kid, 'cn:deepseek-v4.1-flash', 10, 5, 0.1, realm='cn')
        with mock.patch.object(stats, '_fetch_upstream_stats', return_value=PAYLOAD):
            g = stats.summary(realm='global', user=_USER)
        self.assertEqual(g['today_requests'], 334)

    # ── 4. 开关本身 ────────────────────────────────────────────
    def test_has_local_usage_transitions(self) -> None:
        self.assertFalse(stats._has_local_usage(None))
        db.bump_usage(self.kid, 'cn:x', 1, 1, 0.0, realm='cn')
        self.assertTrue(stats._has_local_usage(None))

    def test_failure_only_logs_suppress_fallback(self) -> None:
        """只有失败请求（零 token，不进用量表）时也不能回退。

        那是「网关正在被使用」的证据；若只看 usage_daily，回退会把失败数也覆盖掉，
        等于抹掉现场。
        """
        now = int(time.time())
        db.execute(
            'INSERT INTO request_logs(ts,key_id,ip,status,realm) VALUES(?,?,?,?,?)',
            (now, self.kid, '1.2.3.4', 503, 'cn'),
        )
        self.assertTrue(stats._has_local_usage(None), '有请求日志就算网关被用过')
        with mock.patch.object(stats, '_fetch_upstream_stats') as m:
            s = stats.summary(user=_USER)
            h = stats.hourly(user=_USER)
        m.assert_not_called()
        self.assertIsNone(s['upstream'])
        self.assertEqual(s['failures']['today_5xx'], 1, '失败数不能被回退抹掉')
        self.assertEqual(sum(x['failed'] for x in h), 1, '小时失败数同样保留')

    def test_fetch_failure_is_silent_and_no_fallback(self) -> None:
        """上游取不到（None）时统计页照常返回本地口径的 0，不报错。"""
        with mock.patch.object(stats, '_fetch_upstream_stats', return_value=None):
            s = stats.summary(user=_USER)
        self.assertIsNone(s['upstream'])
        self.assertEqual(s['today_requests'], 0)


if __name__ == '__main__':
    unittest.main()
