import os

import torch
from ultralytics import RTDETR

if __name__ == "__main__":
    # Reduce allocator fragmentation on long CUDA runs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    model = RTDETR("rtdetr-WTConv.yaml")
    print(model)
    model.train(
        data="./datasets/CST_AntiUAV/frhybrid/dataset_nohy.yaml",
        imgsz=640,
        epochs=40,
        batch=4,
        workers=4,
        amp=True,
        cache=False,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    
    