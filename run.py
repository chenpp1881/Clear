import logging
import random
import numpy as np
import torch
import argparse
from data_utils import load_data
from train import CodeBERTTrainer, TransformerTrainer, ClipmlmTrainer, \
    ClipmlmClassifierTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--dataset', type=str, default='RE',
                        choices=['RE', 'TD', 'IO'])

    # clip stage args
    parser.add_argument('--epoch_clip', type=int, default=100)
    parser.add_argument('--batch_size_clip', type=int, default=32)
    parser.add_argument('--lr_clip', type=float, default=1e-5)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--mlmloss', type=float, default=0.1)
    parser.add_argument('--maskVV', action='store_true')
    parser.add_argument('--maskVN', action='store_true')

    # classifier stage args
    parser.add_argument('--epoch_cla', type=int, default=20)
    parser.add_argument('--batch_size_cla', type=int, default=32)
    parser.add_argument('--lr_2', type=float, default=1e-5)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--savepath', type=str, default='./Results/mlm')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_file', type=str, default=None)
    parser.add_argument('--train_clip', action='store_false')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # parse agrs
    args = parse_args()
    logger.info(vars(args))

    # select device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device is %s', args.device)

    # set seed
    set_seed(args.seed)

    # get data
    dataset_pos, train_dataset, test_dataset = load_data(args)

    # cla_trainer = CodeBERTTrainer(args)
    # cla_trainer = TransformerTrainer(args)
    # cla_trainer.train(train_dataset, test_dataset)

    # train or classify
    if args.train_clip:
        trainer = ClipmlmTrainer(args)
        trainer.train(train_dataset, dataset_pos)
        cla_trainer = ClipmlmClassifierTrainer(args)
        cla_trainer.train(train_dataset, test_dataset)


