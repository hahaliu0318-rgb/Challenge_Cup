import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'router'))
from fastapi.testclient import TestClient
from PIL import Image
from app import main as gateway
from app.worker_manager import JobCancelled
from client import sample_request
from configure import build,gpu_list
from check_release import inspect,release_files
from preflight import weights_present
from prepare_model_views import create_view

class PackagingTests(unittest.TestCase):
    def test_public_content_checks(self):
        report=inspect()
        self.assertEqual(report['errors'],[])
        self.assertFalse(report['final_submission_ready'])

    def test_private_config_excluded(self):
        names=[p.relative_to(ROOT).as_posix() for p in release_files()]
        self.assertNotIn('router/config/router.yaml',names)
        self.assertFalse(any(name.startswith(('external/','runtime/')) for name in names))

    def test_samples_paths_and_order(self):
        single=sample_request('small_vqa')
        self.assertTrue(Path(single['image']).is_absolute())
        pair=sample_request('change_pair')
        self.assertEqual([Path(p).name for p in pair['image']],['before.png','after.png'])

    def test_gpu_validation(self):
        self.assertEqual(gpu_list('1,2',2),[[1,2]])
        for value in ('1,1','-1,2','0'):
            with self.assertRaises(ValueError): gpu_list(value,2)

    def test_config_paths_and_worker_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config,models=build(root/'assets',root/'envs',root/'external',[[0]],[[0,1]],[[1,2]],root=root)
            self.assertEqual(config['server']['host'],'127.0.0.1')
            self.assertEqual(models['models']['zoomsearch']['min_free_mib'],[23552,9728])
            self.assertTrue(models['models']['zoomsearch']['env']['ZOOM_MODEL_PATH'].endswith('llava-onevision-qwen2-7b-ov'))

    def test_model_file_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'p':'missing.safetensors'}}))
            self.assertFalse(weights_present(root))
            (root/'missing.safetensors').write_bytes(b'X'*2048)
            self.assertTrue(weights_present(root))

    def test_model_view_does_not_modify_weights_or_original_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';vision=root/'vision';target=root/'view'
            source.mkdir();vision.mkdir()
            original=json.dumps({'mm_vision_tower':'old'})
            (source/'config.json').write_text(original)
            (vision/'config.json').write_text('{}')
            (source/'weights.safetensors').write_bytes(b'fixture')
            try: create_view(source,target,vision)
            except OSError: self.skipTest('Symlink creation unavailable on this host')
            self.assertEqual((source/'config.json').read_text(),original)
            self.assertTrue((target/'weights.safetensors').is_symlink())
            self.assertEqual(json.loads((target/'config.json').read_text())['mm_vision_tower'],str(vision.absolute()))
            with self.assertRaises(ValueError): create_view(source,target,vision)

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.image=root/'small.png';Image.new('RGB',(512,512),'white').save(self.image)
        self.big=root/'large.png';Image.new('RGB',(1025,10),'white').save(self.big)
        config={'runtime':{'state_dir':str(root/'state'),'log_dir':str(root/'logs'),'jobs_db':str(root/'state/jobs.sqlite3'),'allowed_image_roots':[str(root)],'low_resolution_max_edge':1024},'models':{}}
        self.patcher=patch.object(gateway,'load_config',return_value=config);self.patcher.start()
        self.client_context=TestClient(gateway.app);self.client=self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None,None,None);self.patcher.stop()
        # Close app-created log handles so temporary files can be removed on Windows.
        import logging
        for handler in logging.root.handlers[:]:
            if isinstance(handler,logging.FileHandler): handler.close();logging.root.removeHandler(handler)
        self.temp.cleanup()

    def request(self,task='vqa',large=False):
        return {'image':str(self.big if large else self.image),'text':'What is shown?','task_type':task}

    def test_health_and_preview_all_routes_without_models(self):
        self.assertEqual(self.client.get('/healthz').status_code,200)
        for task,large,expected in [('vqa',False,'qwen'),('caption',True,'geollava'),('color',True,'zoomsearch')]:
            response=self.client.post('/v1/routes/preview',json=self.request(task,large))
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['route']['worker'],expected)

    def test_async_and_sync_result_contract_with_fake_worker(self):
        async def invoke(*args,**kwargs):
            await kwargs['status_callback']('loading_model')
            await kwargs['status_callback']('running')
            return {'answer':'fixture answer','load_seconds':0,'inference_seconds':0,'worker_gpus':[]}
        gateway.app.state.manager.invoke=invoke
        response=self.client.post('/v1/jobs',json=self.request())
        self.assertEqual(response.status_code,202)
        job=response.json()['job_id']
        for _ in range(100):
            final=self.client.get('/v1/jobs/'+job).json()
            if final['status']=='succeeded': break
            time.sleep(.01)
        self.assertEqual(final['result']['raw_answer'],'fixture answer')
        self.assertIn('validation_sec',final['result']['timing'])
        short=self.client.post('/v1/infer?trace=false',json=self.request()).json()
        self.assertEqual(short,{'answer':'fixture answer'})

    def test_waiting_cancel_with_fake_worker(self):
        async def invoke(*args,**kwargs):
            await kwargs['status_callback']('queued_waiting_gpu')
            while not kwargs['cancel_event'].is_set(): await asyncio.sleep(.01)
            raise JobCancelled('test cancellation')
        gateway.app.state.manager.invoke=invoke
        job=self.client.post('/v1/jobs',json=self.request()).json()['job_id']
        for _ in range(100):
            state=self.client.get('/v1/jobs/'+job).json()['status']
            if state=='queued_waiting_gpu': break
            time.sleep(.01)
        self.assertEqual(self.client.delete('/v1/jobs/'+job).status_code,200)
        for _ in range(100):
            state=self.client.get('/v1/jobs/'+job).json()['status']
            if state=='cancelled': break
            time.sleep(.01)
        self.assertEqual(state,'cancelled')

if __name__=='__main__': unittest.main()
