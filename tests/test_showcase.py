"""Check showcase evidence and replay routes without loading a model."""
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'router'))
sys.path.insert(0,str(ROOT/'scripts'))
from app.image_probe import validate_and_probe_images
from app.routing import choose_route
from app.task_classifier import classify_by_rules
from client import sample_request

class ShowcaseTests(unittest.TestCase):
    def cases(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'examples/showcase').glob('*.json'))]

    def test_three_datasets_and_evidence_types(self):
        cases=self.cases()
        self.assertEqual({r['dataset'] for r in cases},{'VRSBench','XLRS-Bench-lite','MME-RealWorld-RS'})
        self.assertEqual(sum(r['historical_evidence_type']=='unified_gateway_request' for r in cases),1)
        self.assertTrue(all(r['gateway_rerun_this_turn'] is False for r in cases))
        self.assertTrue(all(r['raw_output'].strip().lower()==r['reference_answer'].strip().lower() for r in cases))

    def test_original_images_and_hashes(self):
        for record in self.cases():
            meta=record['image'];path=ROOT/meta['file']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),meta['sha256'])
            self.assertEqual(path.stat().st_size,meta['bytes'])
            with Image.open(path) as image:
                self.assertEqual(image.size,(meta['width'],meta['height']))
                self.assertEqual(image.format,meta['format'])
                image.verify()

    def test_replay_prompt_and_routes(self):
        for record in self.cases():
            req=sample_request(record['sample_command'])
            self.assertEqual(req['text'],record['replay_request']['text'])
            for choice in record['original_input']['options'] or []:
                self.assertIn(choice,req['text'])
            _,meta=validate_and_probe_images([req['image']],[ROOT/'examples'])
            task=classify_by_rules(req['text'],1,req['task_type'])
            self.assertEqual(choose_route(task,meta).worker,record['expected_worker'])

    def test_readme_has_exactly_three_main_sections_and_valid_local_links(self):
        text=(ROOT/'README.md').read_text(encoding='utf-8')
        self.assertEqual(re.findall(r'^## (.+)$',text,re.M),['一、示例展示','二、使用者如何进行推理','三、当前版本的复现流程'])
        targets=re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'src="([^"]+)"',text)
        for value in targets:
            if value.startswith(('https://','http://','#')):continue
            self.assertTrue((ROOT/value.split('#')[0]).exists(),value)

if __name__=='__main__':unittest.main()
