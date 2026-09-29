"""Local Ouro viewer server. Standard-library HTTP; one serialized GPU worker."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
import traceback
from urllib.parse import urlparse,parse_qs
import uuid
from ouro_viewer_limits import DEFAULT_NEW_TOKENS,MAX_NEW_TOKENS,MAX_SEQUENCE_TOKENS,READOUT_WINDOW,window_bounds

ROOT=Path(__file__).resolve().parent
WEB=ROOT/'ouro-viewer'
DEFAULT_DATA=ROOT/'results/ouro-viewer'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False))
    temporary.replace(path)


class Store:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def path(self,run_id):
        if not isinstance(run_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',run_id):
            raise ValueError('Invalid run ID')
        return self.root/'runs'/run_id

    def get(self,run_id):
        path=self.path(run_id)/'run.json'
        if not path.exists():raise FileNotFoundError('This saved run was not found')
        return json.loads(path.read_text())

    def save(self,run):
        path=self.path(run['id']);path.mkdir(parents=True,exist_ok=True)
        write_json(path/'run.json',run)
        write_json(path/'meta.json',{k:run[k] for k in ['id','title','kind','created_at','input','generation']}
            |dict(token_count=len(run['tokens']),layer_count=len(run['layers']),has_inventory=any('inventory' in x for x in run['layers'])))

    def list(self):
        entries=[]
        for path in (self.root/'runs').glob('*/meta.json'):
            try:entries.append(json.loads(path.read_text()))
            except (OSError,ValueError):continue
        return sorted(entries,key=lambda r:(r['kind']=='research',r['created_at']),reverse=True)

    def window(self,run_id,start):
        run=self.get(run_id);start,end=window_bounds(start,len(run['tokens']))
        current=run.get('readout_window',dict(start=0,end=len(run['tokens']),size=READOUT_WINDOW,total=len(run['tokens'])))
        if current['start']==start:
            return dict(kind='window',run_id=run_id,window=current,layers=run['layers'],vocabulary=run['vocabulary'])
        path=self.path(run_id)/f'window-{start:06d}.json'
        if not path.exists():raise FileNotFoundError('This token window has not been computed yet')
        return json.loads(path.read_text())

    def extras(self,run_id,start=0):
        directory=self.path(run_id)
        if not (directory/'run.json').exists():raise FileNotFoundError('Saved run not found')
        traces={};inventories={}
        for path in directory.glob('trace-*.json'):
            value=json.loads(path.read_text())
            if value.get('position_start',0)==start:traces[value['word'].strip().lower()]=value
        for path in directory.glob('cell-*.json'):
            value=json.loads(path.read_text());inventories[f'{value["site"]}:{value["position"]}']=value
        return dict(traces=traces,inventories=inventories)


def validate_request(value):
    if not isinstance(value,dict):raise ValueError('Expected a JSON object')
    kind=value.get('kind','analyze')
    if kind not in ['analyze','trace','inventory','intervene','window']:raise ValueError('Unknown operation')
    if kind=='analyze':
        prompt=value.get('prompt','')
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Enter a prompt to analyze')
        if len(prompt)>16000:raise ValueError('Prompt is too long (maximum 16,000 characters)')
        if value.get('mode','raw') not in ('raw','chat'):raise ValueError('Choose raw text or chat')
        for name in ['system','prefill']:
            if not isinstance(value.get(name,''),str) or len(value.get(name,''))>4000:raise ValueError('Invalid '+name)
        n=value.get('max_new_tokens',DEFAULT_NEW_TOKENS)
        if type(n) is not int or not 0<=n<=MAX_NEW_TOKENS:raise ValueError(f'Output length must be between 0 and {MAX_NEW_TOKENS:,} tokens')
        value=dict(value,max_new_tokens=n)
    else:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',str(value.get('run_id',''))):raise ValueError('Select a saved run')
    if kind=='trace':
        word=value.get('word','')
        if not isinstance(word,str) or not 0<len(word.strip())<=80:raise ValueError('Enter one concept to trace')
    if kind in ['trace','window']:
        start=value.get('start',0)
        window_bounds(start,MAX_SEQUENCE_TOKENS)
        value=dict(value,start=start)
    if kind=='inventory':
        for key,limit in [('site',192),('position',MAX_SEQUENCE_TOKENS)]:
            if type(value.get(key)) is not int or not 0<=value[key]<limit:raise ValueError('Invalid '+key)
    if kind=='intervene':
        if value.get('operation') not in ['transfer','erase']:raise ValueError('Choose transfer or erase')
        for key in ['source','destination']:
            if key=='destination' and value['operation']=='erase':continue
            if not isinstance(value.get(key),str) or not 0<len(value[key].strip())<=80:raise ValueError('Enter a '+key+' concept')
        loops=value.get('loops',[2,3,4]);band=value.get('band',[24,40])
        if not isinstance(loops,list) or not loops or any(type(x) is not int or x not in [1,2,3,4] for x in loops):raise ValueError('Select at least one loop')
        if not isinstance(band,list) or len(band)!=2 or any(type(x) is not int for x in band) or not 0<=band[0]<=band[1]<=46:raise ValueError('Use a layer band from 0 to 46')
        value=dict(value,loops=loops,band=band)
    return dict(value,kind=kind)


class Jobs:
    def __init__(self,store,*,read_only=False,gpu='0'):
        self.store=store;self.read_only=read_only;self.gpu=gpu
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='ouro-gpu')
        self.lock=threading.RLock();self.jobs={};self.engine=None;self.engine_state='cold'

    def submit(self,payload):
        if self.read_only:raise PermissionError('Live analysis is disabled in this viewer')
        payload=validate_request(payload)
        with self.lock:
            if sum(j['state'] in ['queued','running'] for j in self.jobs.values())>=4:raise ValueError('The analysis queue is full. Wait for a run to finish.')
            job_id=uuid.uuid4().hex[:16]
            job=dict(id=job_id,kind=payload['kind'],state='queued',phase='Waiting for GPU',progress=0,created_at=now(),cancel=threading.Event())
            self.jobs[job_id]=job
        self.pool.submit(self.work,job_id,payload)
        return self.get(job_id)

    def get(self,job_id):
        with self.lock:
            if job_id not in self.jobs:raise FileNotFoundError('Job not found')
            return {k:v for k,v in self.jobs[job_id].items() if k!='cancel'}

    def cancel(self,job_id):
        with self.lock:
            if job_id not in self.jobs:raise FileNotFoundError('Job not found')
            self.jobs[job_id]['cancel'].set()

    def work(self,job_id,payload):
        job=self.jobs[job_id]
        def update(phase,progress,**extra):
            if job['cancel'].is_set():raise InterruptedError('Analysis cancelled')
            with self.lock:job.update(phase=phase,progress=progress,**extra)
        try:
            with self.lock:job.update(state='running',started_at=now())
            update('Loading Ouro and its lenses',.01)
            if self.engine is None:
                self.engine_state='loading'
                from ouro_viewer_engine import Engine
                self.engine=Engine(self.store)
                self.engine_state='ready'
            update('Preparing analysis',.04)
            result=self.engine.execute(payload,update)
            with self.lock:job.update(state='complete',phase='Complete',progress=1,result=result,finished_at=now())
        except InterruptedError:
            with self.lock:job.update(state='cancelled',phase='Cancelled',finished_at=now())
        except Exception as error:
            traceback.print_exc()
            if self.engine is None:self.engine_state='cold'
            with self.lock:job.update(state='failed',phase='Could not complete analysis',error=str(error),finished_at=now())


def handler_for(store,jobs):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):
            if args and '/api/jobs/' not in str(args[0]):super().log_message(format,*args)

        def send(self,body,status=200,content_type='application/json; charset=utf-8'):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-cache');self.send_header('X-Content-Type-Options','nosniff')
            if 'gzip' in self.headers.get('Accept-Encoding','') and len(body)>1500:
                body=gzip.compress(body,compresslevel=3);self.send_header('Content-Encoding','gzip');self.send_header('Vary','Accept-Encoding')
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass

        def fail(self,error):
            code=404 if isinstance(error,FileNotFoundError) else 403 if isinstance(error,PermissionError) else 400
            self.send(dict(error=str(error)),code)

        def do_GET(self):
            parsed=urlparse(self.path);path=parsed.path
            try:
                if path=='/api/status':
                    self.send(dict(model='Ouro-2.6B',loops=4,layers_per_loop=48,fit_prompts=1000,
                        live=not jobs.read_only,engine=jobs.engine_state,gpu=jobs.gpu,
                        device=jobs.engine.device.type if jobs.engine else 'auto',max_new_tokens=MAX_NEW_TOKENS,
                        default_new_tokens=DEFAULT_NEW_TOKENS,readout_window=READOUT_WINDOW));return
                if path=='/api/runs':self.send(store.list());return
                if path.startswith('/api/runs/') and path.endswith('/extras'):
                    start=int(parse_qs(parsed.query).get('start',['0'])[0])
                    self.send(store.extras(path[len('/api/runs/'):-len('/extras')],start));return
                match=re.fullmatch(r'/api/runs/([a-zA-Z0-9_-]{1,64})/windows/(\d+)',path)
                if match:self.send(store.window(match[1],int(match[2])));return
                if path.startswith('/api/runs/'):
                    run_id=path.removeprefix('/api/runs/');file=store.path(run_id)/'run.json'
                    if not file.is_file():raise FileNotFoundError('Saved run not found')
                    self.send(file.read_bytes());return
                if path.startswith('/api/jobs/'):self.send(jobs.get(path.removeprefix('/api/jobs/')));return
                files={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/app.css':'app.css','/favicon.svg':'favicon.svg'}
                if path not in files:raise FileNotFoundError('Page not found')
                file=WEB/files[path]
                content_type=mimetypes.guess_type(str(file))[0] or 'application/octet-stream'
                self.send(file.read_bytes(),content_type=content_type+'; charset=utf-8')
            except (ValueError,FileNotFoundError,PermissionError) as error:self.fail(error)

        def do_POST(self):
            try:
                origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc!=self.headers.get('Host'):raise PermissionError('Requests must come from this viewer')
                if not self.headers.get('Content-Type','').startswith('application/json'):raise ValueError('Use application/json')
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=65536:raise ValueError('Invalid request size')
                value=json.loads(self.rfile.read(length))
                if self.path=='/api/jobs':self.send(jobs.submit(value),202)
                elif self.path.startswith('/api/cancel/'):
                    jobs.cancel(self.path.removeprefix('/api/cancel/'));self.send(dict(cancelled=True))
                else:raise FileNotFoundError('Endpoint not found')
            except (ValueError,FileNotFoundError,PermissionError) as error:self.fail(error)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8769)
    parser.add_argument('--gpu',choices=['0','1','2','3','8'],default='0')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--read-only',action='store_true')
    args=parser.parse_args();os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ['OURO_VIEWER_DEVICE']=args.device
    # Growing KV tensors otherwise leave many differently sized cached blocks.
    if 'PYTORCH_ALLOC_CONF' not in os.environ and 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='2'
    store=Store(args.data);jobs=Jobs(store,read_only=args.read_only,gpu=args.gpu)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(store,jobs))
    print(f'Ouro viewer: http://127.0.0.1:{args.port} · GPU {args.gpu} · {len(store.list())} saved runs',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();jobs.pool.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__':main()
