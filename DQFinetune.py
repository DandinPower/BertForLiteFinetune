import json
import multiprocessing
import os
import torch
from torch import nn
import pandas as pd # 引用套件並縮寫為 pd  
from models.BertModel import *
from models.Memory import IOCounter,MemoryCounter

def load_pretrained_model(pretrained_dir, num_hiddens, ffn_num_hiddens,num_heads, num_layers, dropout, max_len, devices):
    data_dir = pretrained_dir
    data_dir = 'data/bert.base.torch/'
    #讀取vocab
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    #加載預訓練模組
    bert = BERTModel(vocab_size = len(vocab), num_hiddens = num_hiddens,ffn_num_hiddens = ffn_num_hiddens, num_heads = num_heads,num_layers = num_layers, dropout = dropout, max_len = max_len,norm_shape = [768],ffn_num_input = 768)
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab

def load_finetune_model(path, num_hiddens, ffn_num_hiddens,num_heads, num_layers, dropout, max_len, devices):
    data_dir = 'data/bert.base.torch/'
    #讀取vocab
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    model = BERTModel(vocab_size = len(vocab), num_hiddens = num_hiddens,ffn_num_hiddens = ffn_num_hiddens, num_heads = num_heads,num_layers = num_layers, dropout = dropout, max_len = max_len,norm_shape = [768],ffn_num_input = 768)
    bert = BERTClassifier(model)
    bert.load_state_dict(torch.load(path))
    return bert, vocab

class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(768, 2)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, datasetPath, max_len, vocab,train,splitRate):
        self.max_len = max_len
        self.labels = []
        self.vocab = vocab
        self.all_tokens_ids = []
        self.all_segments = []
        self.valid_lens = []
        self.path = datasetPath
        self.train = train
        self.splitRate = splitRate
        self.iocounter = IOCounter()
        self.memorycounter = MemoryCounter()
        self.Preprocess()

    #將資料做預處理
    def Preprocess(self):
        texts,self.labels = self.ReadDataset()
        texts = [self.TruncatePairOfTokens(text)for text in texts]
        newTexts,newSegments = [],[]
        for text in texts:
            tokens,segments = self.GetTokensAndSegments(text)
            newTexts.append(tokens)
            newSegments.append(segments)
        self.PadBertInput(newTexts, newSegments)

    #讀取dataset
    def ReadDataset(self):
        df = pd.read_csv(self.path)
        labels = []
        texts = []
        for i in range(len(df.Stars.values)):
            text = df.Text.values[i]
            label = df.Stars.values[i]
            if (type(text) != str): continue
            if label >= 4:
                labels.append(1)
            else:
                labels.append(0)
            texts.append(text.strip().lower().split(' '))
        trainLen = int(len(df.Text.values) * self.splitRate) 
        if (self.train):
            texts = texts[0:trainLen]
            labels = labels[0:trainLen]
        else:
            texts = texts[trainLen:]
            labels = labels[trainLen:]
        return texts,labels

    def GetTokensAndSegments(self,tokensA, tokensB=None):
        tokens = ['<cls>'] + tokensA + ['<sep>']
        # 0 and 1 are marking segment A and B, respectively
        segments = [0] * (len(tokensA) + 2)
        if tokensB is not None:
            tokens += tokensB + ['<sep>']
            segments += [1] * (len(tokensB) + 1)
        return tokens, segments

    #給<CLS>,<SEP>,<SEP>保留位置
    def TruncatePairOfTokens(self, tokens):   
        while len(tokens) > self.max_len - 3:
            tokens.pop()
        return tokens

    #進行padding
    def PadBertInput(self,texts,segments):
        texts = self.vocab[texts]
        for (text,segment) in zip(texts,segments):
            paddingText = torch.tensor(text + [self.vocab['<pad>']] * (self.max_len - len(text)), dtype=torch.long)
            self.all_tokens_ids.append(paddingText)
            self.all_segments.append(torch.tensor(segment + [0] * (self.max_len - len(segment)), dtype=torch.long))
            #valid_lens不包括<pad>
            self.valid_lens.append(torch.tensor(len(text), dtype=torch.float32))

    def __getitem__(self, idx):
        return (self.all_tokens_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_tokens_ids)

def Finetune():
    devices = d2l.try_all_gpus()
    batch_size, max_len= 32, 512
    train_test_rate = 0.9
    lr, num_epochs = 1e-4, 3
    model_save_path = "models/bert_finetune.model"
    dataset_path = 'dataset/reviews_medium.csv'
    print("Loading Pretraining Model...")
    #重新微調
    bert, vocab = load_pretrained_model('bert.small', num_hiddens=768, ffn_num_hiddens=3072, num_heads=12,num_layers=12, dropout=0.1, max_len=512, devices=devices)
    net = BERTClassifier(bert)
    #讀取訓練過的
    #bert, vocab = load_finetune_model(model_save_path, num_hiddens=256, ffn_num_hiddens=512, num_heads=4,num_layers=2, dropout=0.1, max_len=512, devices=devices)
    #net = bert
    print("Loading Train Dataset...")
    trainDataset = YelpDataset(dataset_path,max_len,vocab,True,train_test_rate)
    train_iter = torch.utils.data.DataLoader(trainDataset, batch_size, shuffle=True)
    print("Loading Test Dataset...")
    testDataset = YelpDataset(dataset_path,max_len,vocab,False,train_test_rate)
    test_iter = torch.utils.data.DataLoader(testDataset, batch_size)
    print("training...")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    finetune_train(net, train_iter, test_iter, loss, trainer, num_epochs, model_save_path,
        devices) 
    torch.save(net.state_dict(), model_save_path)

def DynamicQuantizationFinetune():
    devices = d2l.try_all_gpus()
    batch_size, max_len= 32, 512
    train_test_rate = 0.9
    lr, num_epochs = 1e-4, 3
    model_save_path = "models/bert_finetune_quantization.model"
    dataset_path = 'dataset/reviews_small.csv'
    print("Loading Pretraining Model...")
    #重新微調
    bert, vocab = load_pretrained_model('bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,num_layers=2, dropout=0.1, max_len=512, devices=devices)
    net = BERTClassifier(bert)
    quantized_model = torch.quantization.quantize_dynamic(
        net, {torch.nn.Linear}, dtype=torch.qint8
    )
    #讀取訓練過的
    #bert, vocab = load_finetune_model(model_save_path, num_hiddens=256, ffn_num_hiddens=512, num_heads=4,num_layers=2, dropout=0.1, max_len=512, devices=devices)
    #net = bert
    print("Loading Train Dataset...")
    trainDataset = YelpDataset(dataset_path,max_len,vocab,True,train_test_rate)
    train_iter = torch.utils.data.DataLoader(trainDataset, batch_size, shuffle=True)
    print("Loading Test Dataset...")
    testDataset = YelpDataset(dataset_path,max_len,vocab,False,train_test_rate)
    test_iter = torch.utils.data.DataLoader(testDataset, batch_size)
    print("training...")
    trainer = torch.optim.Adam(quantized_model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    finetune_train(quantized_model, train_iter, test_iter, loss, trainer, num_epochs, model_save_path,
        devices) 
    torch.save(quantized_model.state_dict(), model_save_path)

def CompareParameter():
    devices = d2l.try_all_gpus()
    model_save_path = "models/bert_finetune.model"
    originalBert, _ = load_pretrained_model('bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,num_layers=2, dropout=0.1, max_len=512, devices=devices)
    originalNet = BERTClassifier(originalBert)
    newBert, _ = load_finetune_model(model_save_path, num_hiddens=256, ffn_num_hiddens=512, num_heads=4,num_layers=2, dropout=0.1, max_len=512, devices=devices)
    newNet = newBert 
    #print(newNet.bert.hidden.parameters())
    for parameter in newNet.bert.hidden.parameters():
        print(parameter)
    
    for parameter in originalNet.bert.hidden.parameters():
        print(parameter)

def Inference():
    devices = d2l.try_all_gpus()
    model_save_path = "models/bert_finetune.model"
    dataset_path = 'dataset/reviews_medium.csv'
    batch_size, max_len= 32, 512
    train_test_rate = 0.05
    lr, num_epochs = 1e-4, 3
    newBert, vocab = load_finetune_model(model_save_path, num_hiddens=768, ffn_num_hiddens=3072, num_heads=12,num_layers=12, dropout=0.1, max_len=512, devices=devices)
    quantized_model = torch.quantization.quantize_dynamic(
        newBert, {torch.nn.Linear}, dtype=torch.qint8
    )
    print(f'FP32 Model Size: ',end='')
    print_size_of_model(newBert)
    print(f'INT8 Model Size: ',end='')
    print_size_of_model(quantized_model)
    print("Loading Test Dataset...")
    testDataset = YelpDataset(dataset_path,max_len,vocab,True,train_test_rate)
    test_iter = torch.utils.data.DataLoader(testDataset, batch_size)
    print('testing...')
    test_acc = d2l.evaluate_accuracy_gpu(newBert, test_iter)
    print(f'original test acc {test_acc:.3f}')
    test_acc = d2l.evaluate_accuracy_gpu(quantized_model, test_iter)
    print(f'quantization test acc {test_acc:.3f}')

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

if __name__ == "__main__":
    #Finetune()
    #DynamicQuantizationFinetune()
    #CompareParameter()
    Inference()
