"""Viewer contracts: request validation, persistence, queue safety and HTTP scope."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from ouro_viewer import Store,Jobs,validate_request,handler_for,ThreadingHTTPServer


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.directory=TemporaryDirectory();self.store=Store(Path(self.directory.name))

    def tearDown(self):self.directory.cleanup()

    def fixture(self):
        return dict(id='sample-run',title='<script>literal text</script>',kind='live',created_at='2026-01-01',
                    input=dict(prompt='hello '*1000),generation={},tokens=[dict(id=3,text='<think>')],layers=[])

    def test_store_round_trip_and_listing_keep_literal_text(self):
        run=self.fixture();self.store.save(run)
        self.assertEqual(self.store.get(run['id']),run)
        self.assertEqual(self.store.list()[0]['title'],run['title'])
        self.assertEqual(self.store.list()[0]['token_count'],1)

    def test_identifiers_cannot_escape_run_directory(self):
        for value in ['../model','/etc/passwd','x/y','x%2Fy','a..b','',None]:
            with self.subTest(value=value),self.assertRaises(ValueError):self.store.path(value)

    def test_invalid_lengths_and_modes_are_rejected(self):
        for extra in [dict(max_new_tokens=-1),dict(max_new_tokens=129),dict(max_new_tokens=True),dict(max_new_tokens=1.2),dict(mode='invalid')]:
            with self.subTest(extra=extra),self.assertRaises(ValueError):validate_request(dict(prompt='test',**extra))
        for value in [[],None,dict(prompt=' '),dict(prompt='x'*16001),dict(prompt='hi',system=[])]:
            with self.assertRaises(ValueError):validate_request(value)

    def test_intervention_constraints_precede_model_loading(self):
        valid=dict(kind='intervene',run_id='sample',operation='transfer',source='France',destination='China',loops=[2,3,4],band=[24,40])
        self.assertEqual(validate_request(valid),valid)
        defaults=validate_request({k:v for k,v in valid.items() if k not in ['loops','band']})
        self.assertEqual(defaults['loops'],[2,3,4]);self.assertEqual(defaults['band'],[24,40])
        for extra in [dict(loops=[]),dict(loops=[5]),dict(loops=[True]),dict(band=[40,24]),dict(band=[0,47]),dict(source=''),dict(operation='unknown')]:
            with self.subTest(extra=extra),self.assertRaises(ValueError):validate_request(valid|extra)

    def test_trace_and_inventory_boundaries(self):
        for value in [dict(kind='trace',run_id='a',word=' '),dict(kind='inventory',run_id='a',site=192,position=0),dict(kind='inventory',run_id='a',site=0,position=512)]:
            with self.assertRaises(ValueError):validate_request(value)

    def test_jobs_are_serial_and_cancelled_before_execution(self):
        release=threading.Event();entered=threading.Event();seen=[]
        class Fake:
            device=SimpleNamespace(type='cpu')
            def execute(self,payload,update):
                seen.append(payload['prompt']);entered.set();release.wait(2);update('finishing',.9);return dict(kind='run',run_id='test')
        jobs=Jobs(self.store);jobs.engine=Fake()
        try:
            first=jobs.submit(dict(prompt='first'));self.assertTrue(entered.wait(1))
            second=jobs.submit(dict(prompt='second'));jobs.cancel(second['id']);release.set()
            deadline=time.time()+3
            while time.time()<deadline and jobs.get(second['id'])['state'] in ['queued','running']:time.sleep(.01)
            self.assertEqual(jobs.get(first['id'])['state'],'complete')
            self.assertEqual(jobs.get(second['id'])['state'],'cancelled')
            self.assertEqual(seen,['first'])
        finally:release.set();jobs.pool.shutdown(wait=True)

    def test_http_static_allowlist_gzip_and_origin(self):
        run=self.fixture();self.store.save(run);jobs=Jobs(self.store,read_only=True)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(self.store,jobs));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        root=f'http://127.0.0.1:{server.server_port}'
        try:
            response=urlopen(Request(root+'/api/runs/sample-run',headers={'Accept-Encoding':'gzip'}))
            self.assertEqual(response.headers['Content-Encoding'],'gzip');self.assertEqual(json.loads(gzip.decompress(response.read())),run)
            for path in ['/model/config.json','/../model/config.json','/api/runs/not-found']:
                with self.assertRaises(HTTPError) as caught:urlopen(root+path)
                self.assertEqual(caught.exception.code,404)
            req=Request(root+'/api/jobs',data=b'{"prompt":"hello"}',headers={'Content-Type':'application/json','Origin':'https://unrelated.example'})
            with self.assertRaises(HTTPError) as caught:urlopen(req)
            self.assertEqual(caught.exception.code,403)
            req=Request(root+'/api/jobs',data=b'{"prompt":"hello"}',headers={'Content-Type':'application/json'})
            with self.assertRaises(HTTPError) as caught:urlopen(req)
            self.assertEqual(caught.exception.code,403)
            with self.assertRaises(HTTPError) as caught:urlopen(Request(root+'/api/jobs',data=b'prompt=hi'))
            self.assertEqual(caught.exception.code,400)
        finally:server.shutdown();server.server_close();thread.join();jobs.pool.shutdown(wait=True)


if __name__=='__main__':unittest.main()
