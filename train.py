import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import time
from itertools import cycle
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from clip_mlm import CLIP as CLIPMLM
import transformers
from model import TransformerModel
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def all_metrics(y_true, y_pred, is_training=False):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    return f1.item(), precision.item(), recall.item(), tp.item(), tn.item(), fp.item(), fn.item()

def get_last_resume_file(args):
    files = os.listdir(args.savepath + '/' + args.dataset)
    # 排除文件夹figure
    files = [file for file in files if file.split('.')[-1] == 'pth']
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(files) == 0:
        return None
    return os.path.join(args.savepath, args.dataset, files[-1])

class CodeBERTTrainer():
    def __init__(self, args):
        self.args = args
        model_name = "microsoft/codebert-base"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)

        self.metrics = {'f1': 0, 'precision': 0, 'recall': 0}
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr_2)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        model = torch.nn.DataParallel(model)
        self.model = model.to(args.device)

    def train(self, trainset, devset):
        train_loader = DataLoader(trainset, batch_size=self.args.batch_size_cla, shuffle=True)
        dev_loader = DataLoader(devset, batch_size=self.args.batch_size_cla, shuffle=True)
        for epoch in range(self.args.epoch_cla):
            self.train_epoch(epoch, train_loader)
            logging.info('Epoch %d finished' % epoch)
        self.eval_epoch(epoch, dev_loader)

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (inputs, label) in enumerate(pbar):
            train_encodings = self.tokenizer(inputs, truncation=True, padding=True, return_tensors='pt',
                                             max_length=512)
            ids = train_encodings['input_ids']
            attention_mask = train_encodings['attention_mask']
            ids, label = ids.to(self.args.device), label.to(self.args.device)
            outputs = self.model(input_ids=ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()

            pbar.set_description(f'epoch: {epoch}')
            pbar.set_postfix(index=i, loss=loss.sum().item())

    def eval_epoch(self, epoch, dev_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data, label in tqdm(dev_loader):
                train_encodings = self.tokenizer(data, truncation=True, padding=True, return_tensors='pt')
                ids = train_encodings['input_ids']
                attention_mask = train_encodings['attention_mask']
                ids, label = ids.to(self.args.device), label.to(self.args.device)
                outputs = self.model(input_ids=ids, attention_mask=attention_mask, labels=label)
                logits = outputs.logits
                predicted = torch.argmax(logits, dim=-1)
                all_preds.extend(predicted)
                all_labels.extend(label)

            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, tp, tn, fp, fn = all_metrics(tensor_labels, tensor_preds)
            self.update_best_scores(epoch, f1, precision, recall, tp, tn, fp, fn)

            logger.info(
                'Valid set -f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'
                    .format(f1, precision, recall))
            logger.info('Valid set -tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))

    def update_best_scores(self, epoch, f1, precision, recall, tp, tn, fp, fn):
        if f1 > self.metrics['f1'] or precision > self.metrics['precision'] \
                or recall > self.metrics['recall']:
            self.metrics['f1'] = f1
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.scores2file(epoch, f1, precision, recall, tp, tn, fp, fn)

    def scores2file(self, epoch, f1, precision, recall, tp, tn, fp, fn):
        save_path = self.args.savepath + '/result_record_codebert_' + self.args.dataset + '.csv'
        # add tp tn fp fn to self.matrix type dict
        _record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,

            "epoch": epoch,
            "resume_file": self.args.resume_file,
            "args": self.args
        }
        result_df = pd.DataFrame(_record, index=[0])
        result_df.to_csv(save_path, mode='a', index=False, header=True)
        return

class TransformerTrainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        trymodel = TransformerModel(args)

        self.metrics = {'f1': 0, 'precision': 0, 'recall': 0}
        self.optimizer = optim.AdamW(trymodel.parameters(), lr=args.lr_2)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        model = torch.nn.DataParallel(trymodel)
        self.model = model.to(args.device)

    def train(self, trainset, devset):
        train_loader = DataLoader(trainset, batch_size=self.args.batch_size_cla, shuffle=True, drop_last=True)
        dev_loader = DataLoader(devset, batch_size=self.args.batch_size_cla, shuffle=True, drop_last=True)
        for epoch in range(self.args.epoch_cla):
            self.train_epoch(epoch, train_loader)
            logging.info('Epoch %d finished' % epoch)
        self.eval_epoch(epoch, dev_loader)

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (data, label) in enumerate(pbar):
            train_encodings = self.tokenizer(data, truncation=True, padding=True, return_tensors='pt', max_length=2048)
            ids = train_encodings['input_ids']
            ids, label = ids.to(self.args.device), label.to(self.args.device)
            outputs = self.model(input_ids=ids)
            loss = self.criterion(outputs, label)

            _, predicted = torch.max(outputs.data, dim=1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pbar.set_description(f'epoch: {epoch}')
            pbar.set_postfix(index=i, loss=loss.sum().item())

    def eval_epoch(self, epoch, dev_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data, label in tqdm(dev_loader):
                train_encodings = self.tokenizer(data, truncation=True, padding=True, return_tensors='pt',
                                                 max_length=self.args.max_length)
                ids = train_encodings['input_ids']
                ids, label = ids.to(self.args.device), label.to(self.args.device)
                outputs = self.model(input_ids=ids)
                _, predicted = torch.max(outputs.data, dim=1)
                all_preds.extend(predicted)
                all_labels.extend(label)
            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, tp, tn, fp, fn = all_metrics(tensor_labels, tensor_preds)
            self.update_best_scores(epoch, f1, precision, recall, tp, tn, fp, fn)

            logger.info(
                'Valid set -f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'
                    .format(f1, precision, recall))
            logger.info('Valid set -tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))

    def update_best_scores(self, epoch, f1, precision, recall, tp, tn, fp, fn):
        if f1 > self.metrics['f1'] or precision > self.metrics['precision'] \
                or recall > self.metrics['recall']:
            self.metrics['f1'] = f1
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.scores2file(epoch, f1, precision, recall, tp, tn, fp, fn)

    def scores2file(self, epoch, f1, precision, recall, tp, tn, fp, fn):
        save_path = self.args.savepath + f'/result_record_' + self.args.dataset + '.csv'
        # add tp tn fp fn to self.matrix type dict
        _record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,

            "epoch": epoch,
            "resume_file": self.args.resume_file,
            "args": self.args
        }
        result_df = pd.DataFrame(_record, index=[0])
        result_df.to_csv(save_path, mode='a', index=False, header=True)
        return

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, args, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.args = args

    def forward(self, output1, output2, label):
        assert self.args.maskVV & self.args.maskVN != True
        euclidean_distance = F.pairwise_distance(output1, output2)
        if self.args.maskVV:
            return torch.mean((1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        elif self.args.maskVN:
            return torch.mean((label) * torch.pow(euclidean_distance, 2))
        else:
            return torch.mean((label) * torch.pow(euclidean_distance, 2) +
                              (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                      2))

class ClipmlmTrainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.start_epoch = 0
        self.k = 0

        self.text_seq_len = args.max_length
        clip = CLIPMLM(
            args=args,
            dim_text=512,
            num_text_tokens=50265,
            text_seq_len=self.text_seq_len,
            text_heads=8
        )
        self.optimizer = optim.AdamW(clip.parameters(), lr=args.lr_1)
        self.c_loss = ContrastiveLoss(args=args)

        clip = torch.nn.DataParallel(clip, device_ids=[0, 1])
        self.model = clip.to(args.device)
        if args.resume_file:
            assert os.path.exists(args.resume_file), 'checkpoint not found!'
            logger.info('loading model checkpoint from %s..' % args.resume_file)
            checkpoint = torch.load(args.resume_file)
            clip.load_state_dict(checkpoint['state_dict'], strict=False)
            # self.start_epoch = checkpoint['k'] + 1

    def train(self, all_dataset, p_dataset):
        logging.info(f'Start clipmlm training!')
        all_dataloader = DataLoader(all_dataset, batch_size=self.args.batch_size_1, shuffle=True, drop_last=True)
        p_dataloader = DataLoader(p_dataset, batch_size=self.args.batch_size_1, shuffle=True, drop_last=True)
        for epoch in range(self.start_epoch, self.args.epoch_1 + self.start_epoch):
            self.train_epoch(epoch, all_dataloader, p_dataloader)

    def train_epoch(self, epoch, all_dataloader, p_dataloader):
        self.model.train()

        pbar = tqdm(zip(all_dataloader, cycle(p_dataloader)), total=len(all_dataloader))
        loss_num = 0
        for i, (data, data_p) in enumerate(pbar):
            code, label = data[0], data[1]
            code_p, label_p = data_p[0], data_p[1]

            ids = self.tokenizer(list(code), padding=True, truncation=True, return_tensors='pt',
                                 max_length=self.text_seq_len)['input_ids'].to(
                self.args.device)
            ids_p = self.tokenizer(list(code_p), padding=True, truncation=True, return_tensors='pt',
                                   max_length=self.text_seq_len)['input_ids'].to(self.args.device)
            label = label.to(self.args.device)
            label_p = label_p.to(self.args.device)

            CLS1, CLS2, ssl_loss = self.model(text1=ids,
                                              text2=ids_p,
                                              training_classifier=False)
            loss = self.c_loss(CLS1, CLS2, label & label_p)
            pbar.set_description(f'epoch: {epoch}')
            # loss and step
            pbar.set_postfix(index=i, loss=loss.sum().item())
            loss = self.args.mlmloss * ssl_loss.sum() + loss.sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_num += loss.sum().item()

        if epoch % 19 == 0:
            self.savemodel(epoch)
            self.plot_contracts(all_dataloader.dataset, epoch)

        logger.info(f'epoch:{epoch},loss:{loss_num / len(pbar):.4f}')

    def savemodel(self, k):
        if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset)):
            os.mkdir(os.path.join(self.args.savepath, self.args.dataset))
        torch.save({'state_dict': self.model.state_dict(),
                    'k': k,
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.savepath, self.args.dataset,
                                f'model_{k}.pth'))
        logger.info(f'save:{k}.pth')

    def plot_contracts(self, dataset, k):
        contracts = []
        label = []
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size_1, shuffle=True, drop_last=True)
        for i, data in enumerate(dataloader):
            code, label_ = data[0], data[1]
            ids = self.tokenizer(list(code), padding=True, truncation=True, return_tensors='pt',
                                 max_length=self.text_seq_len)['input_ids'].to(
                self.args.device)
            label_ = label_.to(self.args.device)
            output = self.model(text=ids, label=label_, train=False, return_encodings=True)
            contracts.extend(output.to('cpu').detach().numpy().tolist())
            label.extend(label_.to('cpu').detach().numpy().tolist())

        # pca to 2d
        pca = PCA(n_components=2)
        pca.fit(contracts)
        contracts = pca.transform(contracts)
        plt.scatter(contracts[:, 0], contracts[:, 1], c=label)
        # plt.show()

        # save figure
        if not os.path.exists(os.path.join(self.args.savepath, self.args.dataset, "figure")):
            os.mkdir(os.path.join(self.args.savepath, self.args.dataset, "figure"))
        plt.savefig(os.path.join(self.args.savepath, self.args.dataset, "figure", f'figure_{k}.png'))
        logger.info(f'figure_{k}.png')
        plt.close()

class ClipmlmClassifierTrainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

        self.metrics = {'f1': 0, 'precision': 0, 'recall': 0}
        self.text_seq_len = args.max_length
        clip = CLIPMLM(
            args=args,
            dim_text=512,
            num_text_tokens=50265,
            text_seq_len=self.text_seq_len,
            text_heads=8
        )

        self.optimizer = optim.AdamW(clip.parameters(), lr=args.lr_2)
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        clip = torch.nn.DataParallel(clip, device_ids=[0, 1])
        self.model = clip.to(args.device)
        if args.resume_file:
            assert os.path.exists(args.resume_file), 'checkpoint not found!'
            logger.info('loading model checkpoint from %s..' % args.resume_file)
            checkpoint = torch.load(args.resume_file)
            clip.load_state_dict(checkpoint['state_dict'], strict=False)
            # self.start_epoch = checkpoint['n_epoch'] + 1
        else:
            resume_path = get_last_resume_file(args)
            logger.info('loading model checkpoint from %s..' % resume_path)
            checkpoint = torch.load(resume_path)
            clip.load_state_dict(checkpoint['state_dict'], strict=False)
            # self.start_epoch = checkpoint['n_epoch'] + 1
        self.results_data = []

    def train(self, trainset, devset):
        logging.info('Star clip mlm classifier')
        train_loader = DataLoader(trainset, batch_size=self.args.batch_size_cla, shuffle=True)
        dev_loader = DataLoader(devset, batch_size=self.args.batch_size_cla, shuffle=True)
        for epoch in range(self.args.epoch_cla):
            self.train_epoch(epoch, train_loader)
            logging.info('Epoch %d finished' % epoch)
        self.eval_epoch(dev_loader)
        result_df = pd.DataFrame(self.results_data, columns=['f1', 'precision', 'recall'])
        save_path = self.args.savepath + '/result_record_trymodel_' + self.args.dataset + '.csv'
        result_df.to_csv(save_path, mode='a', index=False, header=True)

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        loss_num = 0.0
        all_labels = []
        all_preds = []
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (inputs, label) in enumerate(pbar):
            token = self.tokenizer(list(inputs), padding=True, truncation=True, return_tensors='pt',
                                   max_length=self.text_seq_len)
            ids = token['input_ids']
            ids, label = ids.to(self.args.device), label.to(self.args.device)
            outputs = self.model(text1=ids,
                                 training_classifier=True)
            loss = self.criterion(outputs, label)

            _, predicted = torch.max(outputs.data, dim=1)
            all_preds.extend(predicted)
            all_labels.extend(label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_num += loss.item()

            pbar.set_description(f'epoch: {epoch}')
            # loss and step
            pbar.set_postfix(index=i, loss=loss.sum().item())

    def eval_epoch(self, dev_loader):
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data, label in tqdm(dev_loader):
                token = self.tokenizer(list(data), padding=True, truncation=True, return_tensors='pt',
                                       max_length=self.text_seq_len)
                ids = token['input_ids']
                ids, label = ids.to(self.args.device), label.to(self.args.device)
                outputs = self.model(text1=ids, training_classifier=True)
                _, predicted = torch.max(outputs.data, dim=1)
                all_preds.extend(predicted)
                all_labels.extend(label)

            tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
            f1, precision, recall, tp, tn, fp, fn = all_metrics(tensor_labels, tensor_preds)
            self.results_data.append([f1, precision, recall])
            logger.info(
                'Valid set -f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'
                    .format(f1, precision, recall))
            logger.info('Valid set -tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))
