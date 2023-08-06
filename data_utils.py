from torch.utils.data import Dataset
from random import choice
from sklearn.model_selection import train_test_split
import logging
import json
logger = logging.getLogger(__name__)

class ContractDataSet(Dataset):
    def __init__(self, data, label):
        super(ContractDataSet, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], int(self.label[idx])

class ContractPositiveDataSet(Dataset):
    def __init__(self, data):
        super(ContractPositiveDataSet, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return choice(self.data), 1

def load_data(args):
    if args.dataset == 'RE':
        p_dataset, train_dataset, test_dataset = load_ree_data()

    elif args.dataset == 'TD':
        p_dataset, train_dataset, test_dataset = load_time_data()

    elif args.dataset == 'IO':
        p_dataset, train_dataset, test_dataset = load_io_data()

    else:
        raise ValueError('No such dataset')
    return p_dataset, train_dataset, test_dataset

def load_io_data():
    all_label = []
    all_data = []
    with open(r'./Data/IO/dataset.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # iter dic
    for data in datas:
        all_data.append(data['code'])
        all_label.append(int(data['label']))

    train_data, test_data, train_label, test_label = split_dataset(all_data,all_label)

    train_dataset = ContractDataSet(train_data, train_label)
    test_dataset = ContractDataSet(test_data,test_label)

    p_data = [train_data[i] for i in range(len(train_label)) if train_label[i] == 1]
    p_dataset = ContractPositiveDataSet(p_data)
    logger.info('IO dataset loaded successfully!')

    return p_dataset, train_dataset, test_dataset

def load_ree_data():
    all_label = []
    all_data = []
    with open(r'./Data/reentrancy/data.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # iter dic
    for file_id, file in datas.items():
        for contract_id, contract in file.items():
            all_data.append(contract['code'])
            all_label.append(contract['lable'])

    train_data, test_data, train_label, test_label = split_dataset(all_data,all_label)

    train_dataset = ContractDataSet(train_data, train_label)
    test_dataset = ContractDataSet(test_data,test_label)

    p_data = [train_data[i] for i in range(len(train_label)) if train_label[i] == 1]
    p_dataset = ContractPositiveDataSet(p_data)
    logger.info('RE dataset loaded successfully!')

    return p_dataset, train_dataset, test_dataset

def load_time_data():
    all_label = []
    all_data = []
    with open(r'./Data/timestamp/data.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # iter dic
    for file_id, file in datas.items():
        for contract_id, contract in file.items():
            all_data.append(contract['code'])
            all_label.append(contract['lable'])

    train_data, test_data, train_label, test_label = split_dataset(all_data,all_label)

    train_dataset = ContractDataSet(train_data, train_label)
    test_dataset = ContractDataSet(test_data,test_label)

    p_data = [train_data[i] for i in range(len(train_label)) if train_label[i] == 1]
    p_dataset = ContractPositiveDataSet(p_data)
    logger.info('TD dataset loaded successfully!')

    return p_dataset, train_dataset, test_dataset

def split_dataset(all_data, all_label, test_size=0.2):

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=test_size, random_state=666)

    logger.info("split dataset into %d train and %d val" % (len(train_data), len(test_data)))
    return train_data, test_data, train_label, test_label