import torch
from torch import nn
from koa_severity.config import ExperimentConfig
from koa_severity.metrics import evaluate_predictions

def test_config_defaults():
    c=ExperimentConfig(); assert c.num_classes==5 and c.image_size==224

def test_metrics_probability_shape():
    r=evaluate_predictions([0,1,2,3,4],[0,1,2,2,4],torch.eye(5),5)
    assert 'macro_roc_auc_ovr' in r

def test_tiny_model_forward():
    m=nn.Linear(4,5); x=torch.randn(3,4); assert m(x).shape==(3,5)
