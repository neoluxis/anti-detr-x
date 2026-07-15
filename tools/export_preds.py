#!/usr/bin/env python3
import sys, os
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo_root, 'ultralytics-git'))
from ultralytics import YOLO

WEIGHT = sys.argv[1] if len(sys.argv)>1 else 'runs/detect/0704/b3/weights/best.pt'
DATA = sys.argv[2] if len(sys.argv)>2 else 'datasets/CST_AntiUAV/cst-sample_train-12000_val-4000_test-4000_seq-100_id-0/data.yaml'
OUT_DIR = sys.argv[3] if len(sys.argv)>3 else 'runs/detect/0704/b3'

print('Using weight:', WEIGHT)
print('Using data:', DATA)
print('Output dir:', OUT_DIR)

model = YOLO(WEIGHT)
# run validation and save JSON
res = model.val(data=DATA, save_json=True, save_dir=OUT_DIR)
print('Validation finished. result:', res)
print('Look for predictions in', OUT_DIR)
