import pickle as pickle
import os
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from inference import *
from omegaconf import OmegaConf
import wandb
import argparse
from load_data import *
from utils import *
from train import *
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "false"

# set fixed random seed
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)               # 시드를 고정해도 함수를 호출할 때 다른 결과가 나오더라..?
    random.seed(seed)
    print('lock_all_seed')


if __name__ =='__main__':
    ## Reset the Memory
    torch.cuda.empty_cache()
    ## parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'../config/{args.config}.yaml')

    ## set seed
    seed_everything(cfg.train.seed)

    ## train
    if cfg.train.train_mode:
        ## wandb login
        wandb.login()
        wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity, name=cfg.wandb.exp_name)

        print('------------------- train start -------------------------')
        train(cfg)

        ## wandb finish
        wandb.finish()

    ## inference
    if cfg.test.test_mode:
        print('--------------------- test start ----------------------')
        test(cfg)

    print('----------------- Finish! ---------------------')