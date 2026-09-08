from __future__ import annotations
import json, os, sys, tempfile, types, unittest
from pathlib import Path
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from marine_3model_experiment import finalize_yolo_stage2_checkpoint
from tools.build_runpod_configs import build
from tools.export_training_artifacts import export_model
from tools.runpod_model_runner import reserve,stream_process
from tools.runpod_orchestrator import run_group
from tools.runpod_release import MODEL_CONFIGS,load_and_validate_config,training_config_sha256
from tools.verify_training_artifacts import verify_torchvision,verify_yolo

COMMIT=os.popen(f'git -C "{ROOT}" rev-parse HEAD').read().strip()

class ConfigurationTests(unittest.TestCase):
    def test_all_official_configs(self):
        for model,name in MODEL_CONFIGS.items(): load_and_validate_config(ROOT/name,model)
    def test_preflight_overrides_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=build(ROOT,Path(tmp)/'configs','unit',True)
            for model,path in paths.items():
                cfg=load_and_validate_config(path,model,allow_preflight=True,allow_output_override=True)
                self.assertEqual(cfg['run']['preflight_sample_limit'],33 if model=='ssd' else 32)
                official=load_and_validate_config(ROOT/MODEL_CONFIGS[model],model)
                for key in ('batch_yolo','batch_frcnn','batch_ssd','imgsz_yolo','imgsz_frcnn','imgsz_ssd','workers','amp'):
                    self.assertEqual(cfg['training'][key],official['training'][key])
    def test_scientific_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=yaml.safe_load((ROOT/MODEL_CONFIGS['yolo']).read_text()); cfg['seed']=7
            path=Path(tmp)/'bad.yaml'; path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(RuntimeError,'seed'): load_and_validate_config(path,'yolo')

class OutputAndOrchestrationTests(unittest.TestCase):
    def test_atomic_duplicate_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json'; reserve(path,{'status':'RUNNING'})
            with self.assertRaisesRegex(RuntimeError,'Duplicate'): reserve(path,{})
    def test_real_subprocess_return_code_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'child.log'; code=stream_process([sys.executable,'-c','print("child-output");raise SystemExit(7)'],ROOT,log,os.environ.copy())
            self.assertEqual(code,7); self.assertIn('child-output',log.read_text())
    def test_sequential_order_real_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp); result=run_group(ROOT,COMMIT,'ok',temp,temp/'env',temp/'control',True,runner=ROOT/'tests/fixtures/fake_runpod_model_runner.py')
            self.assertEqual(result['status'],'VERIFIED_COMPLETE')
            self.assertEqual((temp/'control/order.txt').read_text().splitlines(),['yolo','frcnn','ssd'])
    def test_failure_stops_next_model_and_propagates_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp); control=temp/'control'; control.mkdir(); (control/'fail_frcnn').touch()
            result=run_group(ROOT,COMMIT,'fail',temp,temp/'env',control,True,runner=ROOT/'tests/fixtures/fake_runpod_model_runner.py')
            self.assertEqual(result['return_code'],17); self.assertEqual(result['failed_model'],'frcnn')
            self.assertEqual((control/'order.txt').read_text().splitlines(),['yolo','frcnn'])
    def test_existing_output_is_preserved_by_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'model'; root.mkdir(); (root/'best.pt').write_bytes(b'x'); (root/'prepared_datasets').mkdir(); (root/'prepared_datasets/data').write_bytes(b'secret-data')
            (root/'training_artifact_manifest.json').write_text('{"status":"VERIFIED_COMPLETE"}')
            archive=Path(tmp)/'export.tar.gz'; export_model(root,archive)
            self.assertTrue((root/'prepared_datasets/data').exists()); self.assertTrue(archive.exists())
    def test_unverified_export_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'model'; root.mkdir(); marker=root/'keep.txt'; marker.write_text('keep')
            with self.assertRaisesRegex(RuntimeError,'Unverified'): export_model(root,Path(tmp)/'bad.tar.gz')
            self.assertEqual(marker.read_text(),'keep')

class ArtifactContractTests(unittest.TestCase):
    def test_yolo_selection_publishes_metric_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); (run/'weights').mkdir(); pd.DataFrame({'epoch':[1,2],'metrics/mAP50-95(B)':[.2,.8]}).to_csv(run/'results.csv',index=False)
            (run/'weights/epoch0.pt').write_bytes(b'a'); (run/'weights/epoch1.pt').write_bytes(b'b'); (run/'weights/best.pt').write_bytes(b'native'); (run/'weights/last.pt').write_bytes(b'last')
            best=finalize_yolo_stage2_checkpoint(run); self.assertEqual(best.read_bytes(),b'b')
            report=json.loads((run/'checkpoint_selection.json').read_text()); self.assertEqual(report['selected_epoch_index'],1)
    def test_torchvision_verifier_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in ('frcnn','ssd'):
                out=Path(tmp)/model; root=out/f'runs/seed_42/torchvision/{model}'; root.mkdir(parents=True)
                cfg=load_and_validate_config(ROOT/MODEL_CONFIGS[model],model)
                history=pd.DataFrame([{'epoch':i,'stage':'head' if i<=10 else 'all','map':i/1000,'loss':1.0,'lr':.001} for i in range(1,111)])
                history.to_csv(root/'history.csv',index=False)
                base={'model':{'w':torch.tensor([1])},'cfg':cfg,'epoch':110,'architecture':model,'stage':'all','stage_epoch':100,'best_epoch':110}
                torch.save(base,root/'best.pt')
                last={**base,'optimizer':{},'scheduler':{},'scaler':{},'completed_epoch':110,'next_epoch':111,'training_history':history.to_dict('records'),'training_config_sha256':training_config_sha256(cfg),'dataset_sha256':cfg['provenance']['dataset_sha256'],'split_sha256':cfg['provenance']['split_manifest_sha256'],'git_commit':COMMIT,'seed':42,'experiment_id':'fixture','python_random_state':(),'numpy_random_state':(),'torch_cpu_rng_state':torch.tensor([1],dtype=torch.uint8),'torch_cuda_rng_states':[],'dataloader_generator_state':torch.tensor([1],dtype=torch.uint8),'sampler_state':{}}
                torch.save(last,root/'last.pt'); checkpoints,details=verify_torchvision(model,out,cfg,COMMIT,110)
                self.assertEqual(details['history_rows'],110); self.assertIn('last.pt',checkpoints)
    def test_yolo_verifier_fixture(self):
        fake=types.ModuleType('ultralytics')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp); root=out/'runs/seed_42/yolo'; s1=root/'yolov8s_stage1'; s2=root/'yolov8s_stage2'; (s1/'weights').mkdir(parents=True); (s2/'weights').mkdir(parents=True)
                pd.DataFrame({'epoch':range(1,11),'metrics/mAP50-95(B)':[.1]*10}).to_csv(s1/'results.csv',index=False)
                vals=[.1]*99+[.9]; pd.DataFrame({'epoch':range(1,101),'metrics/mAP50-95(B)':vals}).to_csv(s2/'results.csv',index=False)
                for name,data in [('epoch99.pt',b'z'),('best.pt',b'z'),('last.pt',b'l')]: (s2/'weights'/name).write_bytes(data)
                import hashlib
                h=lambda b:hashlib.sha256(b).hexdigest(); (s2/'checkpoint_selection.json').write_text(json.dumps({'status':'SELECTED_BY_VALIDATION_MAP50_95','stage':'stage2','selected_epoch_index':99,'source_checkpoint':'epoch99.pt','source_checkpoint_sha256':h(b'z'),'best_checkpoint_sha256':h(b'z'),'last_checkpoint_sha256':h(b'l')}))
                fake.YOLO=lambda path: types.SimpleNamespace(model=object(),ckpt={'epoch':99,'train_args':{'seed':42,'imgsz':640,'batch':16,'epochs':100,'name':'yolov8s_stage2'}})
                previous=sys.modules.get('ultralytics'); sys.modules['ultralytics']=fake
                _,details=verify_yolo(out,(10,100)); self.assertEqual(details['stage2_rows'],100)
        finally:
            if 'previous' in locals():
                if previous is None: sys.modules.pop('ultralytics',None)
                else: sys.modules['ultralytics']=previous
    def test_ssd_final_batch_policy(self):
        loader=torch.utils.data.DataLoader(list(range(33)),batch_size=16,drop_last=True)
        self.assertEqual(len(loader),2)

class RealGpuChecks(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable locally; execute on RunPod')
    def test_live_cuda_torchvision_ops(self):
        from tools.runpod_gpu_preflight import torchvision_cuda_ops
        self.assertEqual(torchvision_cuda_ops()['nms'],'PASS')

if __name__=='__main__': unittest.main()
