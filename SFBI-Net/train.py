from argparse import ArgumentParser
from tkinter import N
import torch
import numpy as np
import random
from models.trainer import *


"""
the main function for training the CD networks
"""



def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu

def train(args):
    dataloaders = utils.get_loaders(args)
    if args.add_loss=="True":
        model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    seed_torch(42)
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='BIN_SFC', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=2, type=int) 
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--pixel_edge', default=0, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--pretrain', default=None, type=str)


    parser.add_argument('--net_G', default='base_resnet18', type=str,)
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    parser.add_argument('--sfd_loss_weight', default=0.01, type=float)
    parser.add_argument('--add_loss', default=False, type=str)

    args = parser.parse_args()
    utils.get_device(args)
    print("Current GPU:",args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
