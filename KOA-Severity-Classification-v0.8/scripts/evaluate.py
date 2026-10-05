"""Evaluate a ResNet18 checkpoint on an ImageFolder split."""
from __future__ import annotations
import argparse
import torch
from torch import nn
from torchvision import models
from koa_severity.data import get_dataset_path
from koa_severity.datasets import LoaderConfig, make_imagefolder_loader
from koa_severity.checkpoints import load_checkpoint
from koa_severity.evaluation import predict

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default=None); p.add_argument('--checkpoint',required=True); p.add_argument('--split',default='test'); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--device',default=None); args=p.parse_args()
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'); root=get_dataset_path(args.data)
    ds, loader=make_imagefolder_loader(root,args.split,LoaderConfig(batch_size=args.batch_size),False)
    model=models.resnet18(weights=None); model.fc=nn.Linear(model.fc.in_features,len(ds.classes)); load_checkpoint(args.checkpoint,model,device,strict=True)
    result=predict(model,loader,device); print(result['metrics'])

if __name__=='__main__': main()
