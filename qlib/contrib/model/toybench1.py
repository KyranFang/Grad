import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import torch.optim as optim
from transformers import BertTokenizer, BertModel

import pandas as pd
import numpy as np
import tqdm
import pprint as pp
import sys
import math
import copy

import qlib
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.base import Model

from transformers import BertTokenizer, BertModel
import torch

class NewsEmbedder(nn.Module):
    def __init__(self, model_name = 'bert-base-uncased', d_llm = 768, d_FFN = 768*2, d_out = 512):
        super(NewsEmbedder, self).__init__() 
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(d_llm, d_FFN)
        self.linear2 = nn.Linear(d_FFN, d_out)
    
    def forward(self, x):
        # x : natural language sentence
        x = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            x = self.model(**x)
            x = x.last_hidden_state
        x = x[:, 0, :]
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class TSEmbedder(nn.Module):
    def __init__(self, d_feat=6, d_mlp1=64, d_mlp2=256, d_out=512, device=None):
        super(TSEmbedder, self).__init__()
        self.device = device
        self.MLP = nn.Sequential([
            nn.Linear(d_feat, d_mlp1),
            nn.Linear(d_mlp1, d_mlp2),
            nn.Linear(d_mlp2, d_out)
        ])

    def forward(self, x):
        x = self.MLP(x)
        return x

class ToyFusion(nn.Module):
    def __init__(self, d_ts_in = 6, d_ts_out=512, 
                 d_NLP_out=512, d_FFN=512,
                 news_model_name = 'bert-base-uncased'):
        super(ToyFusion, self).__init__()
        self.d_ts = d_ts_out
        self.d_NLP = d_NLP_out
        self.news_embedder = NewsEmbedder(news_model_name=news_model_name)
        self.ts_embedder = TSEmbedder(d_feat=d_ts_in)
        self.mixer = nn.Linear((d_ts_out+d_NLP_out), d_FFN)
        self.predictor = nn.Linear(d_FFN, 1)

    def forward(self, src_ts, src_news):
        ts_out = self.ts_embedder(src_ts)
        news_out = self.news_embedder(src_news)
        x = torch.cat((ts_out, news_out), dim=0)
        x = self.mixer(x)
        x = self.predictor(x)
        return x

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)

class ToyFusionModel(nn.Module):
    def __init__(self, d_feat, d_mlp1, d_mlp2, n_epochs, 
                 lr, GPU, seed, save_path = 'model/', save_prefix= '', 
                 benchmark = '^npx', beta = 0, train_stop_loss_thred = 0,
                 market = 'nasdaq1--', only_backtest = False):
        super(ToyFusionModel, self).__init__()
        
        self.d_feat = d_feat
        self.d_mlp1 = d_mlp1
        self.d_mlp2 = d_mlp2
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.beta = beta

        self.fitted = False
        self.beta = 5
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.model = ToyFusion(d_ts_in = 6, d_ts_out=512, 
                               d_NLP_out=512, d_FFN=512,
                               news_model_name = 'bert-base-uncased')
        self.train_optim = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)
        self.save_path = save_path
        self.only_backtest = only_backtest
        self.train_stop_loss_thred = train_stop_loss_thred
        
        self.chech_model()

    def check_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized!")
        return None
    
    def load_model(self, param_path):
        try:
            self.model.load_state_dict(torch.load(param_path, map_location=self.device))
            self.fitted = True
        except:
            raise ValueError("Model not found.") 

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True
   
    def loss_fn(self, pred, label):
        mask = ~ torch.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2 
        return torch.mean(loss)
        
    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        
        for data in data_loader:
            data = torch.squeeze(data, dim = 0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 7 factors + 1news + label           
            '''
            
            feature = data[:, :, 0:-2].to(self.device)
            news = data[:, :, -2].to(self.device)
            label = data[:, -1, -1].to(self.device)
            assert not torch.any(torch.isnan(label))
    
            pred = self.model(feature.float(), news)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())
            
            self.train_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optim.step()
            
        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader       
        
    def fit(self, dataset: DatasetH):
        dl_train = dataset.prepare("train", col_set=["feature, label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)

        self.fitted=True
        best_param = None
        best_val_loss = 1
        
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)
            
            print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (step, train_loss, val_loss))
            if best_val_loss > val_loss:
                best_param = copy.deepcopy(self.model.state_dict())
                best_val_loss = val_loss

            if train_loss <= self.train_stop_loss_thred:
                break
        torch.save(best_param, f'{self.save_path}{self.save_prefix}tb1_{self.seed}.pkl')

    def predict(self, dataset: DatasetH, use_pretrained = True):
        if use_pretrained:
            self.load_param(f'{self.save_path}{self.save_prefix}tb1_{self.seed}.pkl')
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        pred_all = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            pred_all.append(pred.ravel())


        pred_all = pd.DataFrame(np.concatenate(pred_all), index=dl_test.get_index())
        # pred_all = pred_all.loc[self.label_all.index]
        # rec = self.backtest()
        return pred_all

        

if __name__ == "__main__":
    sentence = "Hello, how are you?"
