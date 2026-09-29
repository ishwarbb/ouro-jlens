"""Exercise real subprocess scheduling without loading a model or using GPUs."""
import contextlib
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_ouro_workspace as supervisor


WORKER = '''
import argparse,hashlib,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase');p.add_argument('--rank',type=int);p.add_argument('--output',type=Path)
a=p.parse_args()
def save(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(path)
def event(kind):
    with (a.output/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind,rank=a.rank,gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))+'\\n')
if a.phase=='merge':
    (a.output/'merged').touch()
else:
    folder=a.output/'workers'/f'worker-{a.rank}';folder.mkdir(parents=True,exist_ok=True)
    save(folder/'progress.json',dict(pid=os.getpid(),rank=a.rank))
    event('start');time.sleep(.2)
    event('end')
    fingerprint=hashlib.sha256((a.output/'full-manifest.json').read_bytes()).hexdigest()
    save(folder/'complete.json',dict(rank=a.rank,manifest_sha256=fingerprint))
'''

EVALUATOR = '''
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--family');p.add_argument('--gpus');a=p.parse_args()
from pathlib import Path
(Path(a.family)/'evaluated.json').write_text(json.dumps(dict(gpus=a.gpus)))
'''


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / 'run'
        self.output.mkdir()
        (self.output/'full-manifest.json').write_text(json.dumps(dict(workers=8)))
        (self.root/'ouro_full_fit.py').write_text(WORKER)
        (self.root/'workspace_reproduction.py').write_text(EVALUATOR)

    def tearDown(self):
        self.temporary.cleanup()

    def run_supervisor(self):
        argv = ['supervisor', '--output', str(self.output), '--gpus', '0,1,2,3']
        real_sleep = time.sleep
        with patch.object(supervisor, 'ROOT', self.root), patch.object(sys, 'argv', argv), \
                patch.object(supervisor.time, 'sleep', lambda _: real_sleep(.005)), contextlib.redirect_stdout(io.StringIO()):
            supervisor.main()

    def check_run(self):
        events = [json.loads(line) for line in (self.output/'events.jsonl').read_text().splitlines()]
        active, started, peak = {}, set(), 0
        for event in events:
            gpu, rank = event['gpu'], event['rank']
            self.assertIn(gpu, ['0','1','2','3'])
            if event['kind'] == 'start':
                self.assertNotIn(gpu, active)
                self.assertNotIn(rank, started)
                active[gpu] = rank
                started.add(rank)
                peak = max(peak, len(active))
            else:
                self.assertEqual(active.pop(gpu), rank)
        self.assertEqual(started, set(range(8)))
        self.assertEqual(peak, 4)
        self.assertFalse(active)
        self.assertTrue((self.output/'merged').exists())
        self.assertEqual(json.loads((self.output/'evaluated.json').read_text())['gpus'], '0,1,2,3')
        self.assertEqual(json.loads((self.output/'status.json').read_text())['stage'], 'complete')

    def test_eight_shards_never_exceed_four_gpus_and_evaluation_uses_same_pool(self):
        self.run_supervisor()
        self.check_run()

    def test_adopts_surviving_worker_without_duplicate_launch(self):
        # Hold this worker until main() can adopt it, then let it finish normally.
        path = self.root/'ouro_full_fit.py'
        path.write_text(WORKER.replace("event('start');time.sleep(.2)",
            "event('start');time.sleep(.8 if a.rank==0 else .2)"))
        process = subprocess.Popen([sys.executable, str(path), '--phase', 'worker',
            '--rank', '0', '--output', str(self.output)], env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
        try:
            deadline = time.monotonic()+5
            progress = self.output/'workers/worker-0/progress.json'
            while not progress.exists() and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue(progress.exists())
            self.run_supervisor()
            self.check_run()
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)

    def test_duplicate_supervisor_cannot_overwrite_pid_or_launch_workers(self):
        (self.output/'supervisor.pid').write_text('1234\n')
        with (self.output/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(SystemExit, 'Another supervisor'):
                self.run_supervisor()
        self.assertEqual((self.output/'supervisor.pid').read_text(), '1234\n')
        self.assertFalse((self.output/'events.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
