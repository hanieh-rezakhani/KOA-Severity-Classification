"""Train the repository's reproducible ResNet18 baseline."""
from __future__ import annotations
import argparse
import torch
from torch import nn, optim
from torchvision import models
from koa_severity.config import ExperimentConfig
from koa_severity.datasets import LoaderConfig, make_imagefolder_loader
from koa_severity.reproducibility import seed_everything
from koa_severity.training import SupervisedTrainer, TrainConfig
from koa_severity.data import get_dataset_path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data', default=None); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--output',default='checkpoints/resnet18_baseline.pth'); p.add_argument('--device',default=None); p.add_argument('--pretrained',action=argparse.BooleanOptionalAction,default=True); args=p.parse_args()
    seed_everything(2026); cfg=ExperimentConfig(epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.lr)
    root=get_dataset_path(args.data); lc=LoaderConfig(batch_size=cfg.batch_size, weighted_sampling=cfg.weighted_sampling)
    _, train_loader=make_imagefolder_loader(root,'train',lc,True); _, val_loader=make_imagefolder_loader(root,'val',lc,False)
    model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if args.pretrained else None); model.fc=nn.Linear(model.fc.in_features,cfg.num_classes)
    for param in model.parameters(): param.requires_grad=False
    for param in model.fc.parameters(): param.requires_grad=True
    opt=optim.Adam(model.fc.parameters(),lr=cfg.learning_rate)
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    history=SupervisedTrainer(model,nn.CrossEntropyLoss(),opt,TrainConfig(cfg.epochs,cfg.learning_rate,device,args.output)).fit(train_loader,val_loader)
    print(history[-1])

if __name__=='__main__': main()
