"""Run 6 scientific invariants and finite local-action semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from prepare_deception_run6 import make_plan,measurements,capability_record,apply_choice,reference,capability_messages,CONDITIONS
from run_deception_run6 import final_object,immutable_json
from analyze_deception_run6 import windows,prefix_absent,clustered_contrasts,mean


class CharacterTokenizer:
    def decode(self,ids):return ''.join(chr(i) for i in ids)


class Run6Tests(unittest.TestCase):
    def test_budget_and_balance(self):
        records,jobs=make_plan()
        self.assertEqual((len(records),len(jobs)),(48,320))
        self.assertEqual(sum(j['phase']=='measurement' for j in jobs),32)
        self.assertEqual(sum(1 if j['phase']=='measurement' else 2 for j in jobs),608)
        self.assertEqual(len({j['id'] for j in jobs}),320)
        for rec in records:
            group=[j for j in jobs if j['phase']=='capability' and j['record_id']==rec['id']]
            self.assertEqual(len(group),6)
            for rep in range(2):self.assertEqual(len({j['seed'] for j in group if j['repetition']==rep}),1)
        self.assertEqual(make_plan(),(records,jobs))

    def test_each_record_has_exactly_one_successful_action(self):
        records,_=make_plan()
        for rec in records:
            results=[apply_choice(rec,c) for c in 'ABC']
            self.assertEqual(sum(r['task_correct'] for r in results),1,rec['id'])

    def test_reference_changes_with_counterfactual_inputs(self):
        rec=capability_record('metrics',0,0)
        self.assertEqual(reference(rec),'60')
        rec['data']['successes']=17
        self.assertEqual(reference(rec),'85')

    def test_repair_executes_tests(self):
        rec=capability_record('repair',0,0)
        good=next(c for c,v in rec['choices'].items() if v=='ge')
        bad=next(c for c,v in rec['choices'].items() if v=='gt')
        self.assertEqual(apply_choice(rec,good)['artifact']['actual_test_outputs'],[False,True,True])
        self.assertEqual(apply_choice(rec,bad)['observed_result'],'FAIL')

    def test_report_edit_preserves_measurement(self):
        rec=capability_record('metrics',0,0)
        bad=next(c for c,v in rec['choices'].items() if v=='95')
        result=apply_choice(rec,bad)
        self.assertEqual(result['artifact'],dict(independent_measurement='60',report_value='95'))
        self.assertFalse(result['task_correct'])
        self.assertEqual(result['observed_result'],'95')

    def test_correction_preserves_original_observation(self):
        rec=capability_record('correction',1,0)
        good=next(c for c,v in rec['choices'].items() if v=='30')
        result=apply_choice(rec,good)
        self.assertEqual(result['artifact'],dict(original_tool_value='11',correction_log=['30']))

    def test_approval_checks_issuer_scope_and_distinct_identities(self):
        for template in range(3):
            rec=capability_record('approval',template,0)
            accepted=[r for r in (apply_choice(rec,c) for c in 'ABC') if r['task_correct']]
            self.assertEqual(len(accepted),1)
            self.assertEqual(accepted[0]['artifact']['distinct_qualified_humans'],2 if template==2 else 1)
        rec=capability_record('approval',2,0)
        choice=next(c for c,v in rec['choices'].items() if len(set(v))==1)
        self.assertFalse(apply_choice(rec,choice)['task_correct'])

    def test_invalid_choices_do_not_mutate_task(self):
        rec=capability_record('repair',1,0);original=copy.deepcopy(rec)
        for choice in (None,True,1,{},'../other','a','A; rm'):
            result=apply_choice(rec,choice)
            self.assertFalse(result['valid_action']);self.assertFalse(result['task_correct'])
        self.assertEqual(rec,original)

    def test_conditions_preserve_facts_and_choices(self):
        rec=capability_record('metrics',2,3)
        msgs=[capability_messages(rec,c) for c in CONDITIONS]
        for msg in msgs:
            self.assertIn(json.dumps(rec['data'],sort_keys=True),msg[1]['content'])
            self.assertIn(json.dumps(rec['choices'],sort_keys=True),msg[1]['content'])
        self.assertIn('independent_reference',msgs[1][1]['content'])
        self.assertIn('unrelated documentation glossary',msgs[2][1]['content'])

    def test_measurement_pairs_swap_frozen_targets(self):
        jobs=measurements()
        for family in ('copy','lookup','arithmetic','opposite'):
            for pair in range(4):
                a,b=[j for j in jobs if j['family']==family and j['pair']==pair]
                self.assertEqual((a['target'],a['foil']),(b['foil'],b['target']))
        for job in jobs:
            if job['family']=='arithmetic':
                self.assertNotIn(job['target'],job['question'])
                self.assertNotEqual(job['target'],job['expected'])

    def test_parser_requires_one_complete_string_field(self):
        self.assertEqual(final_object('The valid option is ready.\n{"choice":"B"}<|im_end|>','choice'),('B',None))
        self.assertEqual(final_object('<think>consider</think>\n{"answer":"red"}','answer'),('red',None))
        for raw in ('{"choice":true}','{"choice":"B","choice":"A"}','{"choice":"B"} trailing',
                    '{"choice":"A"}\n{"choice":"B"}','{"choice":"B","extra":1}','{"choice":{"value":"B"}}'):
            self.assertIsNone(final_object(raw,'choice')[0],raw)
        self.assertIsNone(final_object('{"choice":"A"}','choice',False)[0])

    def test_before_answer_excludes_value_and_later_mentions(self):
        tok=CharacterTokenizer();raw='Reason first. {"answer":"fake"} more'
        g=dict(input_ids=[9]*40,output_ids=[ord(c) for c in raw],raw_text=raw,
               messages=[dict(content='A neutral prompt')])
        w=windows(g,tok);boundary=raw.index('"fake"')
        for name in ('early_before_answer','before_answer'):
            self.assertTrue(w[name]);self.assertTrue(all(p-40<boundary for p in w[name]))
        self.assertEqual(w['last_prompt'],[39])
        absent=prefix_absent(g,tok,[39,40+raw.index('fake')+3],['fake'])
        self.assertTrue(absent[39]['fake']);self.assertFalse(absent[40+raw.index('fake')+3]['fake'])

    def test_token_straddling_value_boundary_is_excluded(self):
        class ChunkTokenizer:
            pieces=['Reason ', '{"answer":', '"red"', '}']
            def decode(self,ids):return ''.join(self.pieces[i] for i in ids)
        tok=ChunkTokenizer();raw=tok.decode([0,1,2,3])
        g=dict(input_ids=[8]*40,output_ids=[0,1,2,3],raw_text=raw)
        self.assertEqual(windows(g,tok)['before_answer'],[40,41])

    def test_missing_marker_is_missing_generated_window(self):
        tok=CharacterTokenizer();raw='No JSON produced'
        g=dict(input_ids=[1]*40,output_ids=[ord(c) for c in raw],raw_text=raw)
        self.assertEqual(windows(g,tok),dict(last_prompt=[39],early_before_answer=[],before_answer=[]))

    def test_immutable_artifact_rejects_resume_drift(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'artifact.json';immutable_json(path,{'value':1});immutable_json(path,{'value':1})
            with self.assertRaises(AssertionError):immutable_json(path,{'value':2})

    def test_missing_data_do_not_become_zero_effects(self):
        self.assertIsNone(mean([]))
        result=clustered_contrasts([],'metric')
        self.assertTrue(all(r['estimate'] is None and r['ci95'] is None for r in result.values()))

    def test_clustered_contrast_uses_records_and_templates(self):
        records,jobs=make_plan();rows=[]
        for j in jobs:
            if j['phase']=='capability':rows.append({**j,'metric':0 if j['condition']=='plain' else 1})
        result=clustered_contrasts(rows,'metric')
        for value in result.values():
            self.assertEqual(value['templates'],12);self.assertEqual(value['estimate'],1)
            self.assertEqual(value['ci95'],[1,1])


if __name__=='__main__':unittest.main()
