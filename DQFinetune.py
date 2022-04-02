import json
import multiprocessing
import os
import torch
from torch import nn
import pandas as pd 
from models.BertModel import *
from models.Memory import IOCounter,MemoryCounter
PRETRAIN_DIR = 'data/bert.small.torch/'
NUM_HIDDENS = 256  #small:256 base:768
FFN_NUM_HIDDENS = 512 #small:512 base:3072
NUM_HEADS = 4 #small:4 base:12
NUM_LAYERS = 2 #small:2 base:12
DROPOUT = 0.1
MAX_LEN = 512
NORM_SHAPE = [256] #small:[256] base:[768]
FFN_NUM_INPUT = 256 #small:[256] base:[768]

DEVICES = d2l.try_all_gpus()[:2]
BATCH_SIZE = 32
TRAIN_TEST_RATE = 0.9
LR = 1e-4
NUM_EPOCHS = 3
MODEL_SAVE_PATH = "models/bert_finetune.model"
DATASET_PATH = 'dataset/reviews_small.csv'

def load_pretrained_model():
    data_dir = PRETRAIN_DIR
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = BERTModel(vocab_size = len(vocab), num_hiddens = NUM_HIDDENS,ffn_num_hiddens = FFN_NUM_HIDDENS, num_heads = NUM_HEADS,num_layers = NUM_LAYERS, dropout = DROPOUT, max_len = MAX_LEN,norm_shape = NORM_SHAPE,ffn_num_input = FFN_NUM_INPUT)
    bert.load_state_dict(torch.load(os.path.join(data_dir,'pretrained.params')))
    return bert, vocab

def load_finetune_model(path):
    data_dir = PRETRAIN_DIR
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    model = BERTModel(vocab_size = len(vocab), num_hiddens = NUM_HIDDENS,ffn_num_hiddens = FFN_NUM_HIDDENS, num_heads = NUM_HEADS,num_layers = NUM_LAYERS, dropout = DROPOUT, max_len = MAX_LEN,norm_shape = NORM_SHAPE,ffn_num_input = FFN_NUM_INPUT)
    bert = BERTClassifier(model)
    bert.load_state_dict(torch.load(path))
    return bert, vocab

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(NUM_HIDDENS, 2)

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
    print("Loading Pretraining Model...")
    #重新微調
    bert, vocab = load_pretrained_model()
    net = BERTClassifier(bert)
    #讀取訓練過的
    #bert, vocab = load_finetune_model(MODEL_SAVE_PATH)
    #net = bert
    print("Loading Train Dataset...")
    trainDataset = YelpDataset(DATASET_PATH,MAX_LEN,vocab,True,TRAIN_TEST_RATE)
    train_iter = torch.utils.data.DataLoader(trainDataset, BATCH_SIZE, shuffle=True)
    print("Loading Test Dataset...")
    testDataset = YelpDataset(DATASET_PATH,MAX_LEN,vocab,False,TRAIN_TEST_RATE)
    test_iter = torch.utils.data.DataLoader(testDataset, BATCH_SIZE)
    print("training...")
    trainer = torch.optim.Adam(net.parameters(), lr=LR)
    loss = nn.CrossEntropyLoss(reduction='none')
    finetune_train(net, train_iter, test_iter, loss, trainer, NUM_EPOCHS, MODEL_SAVE_PATH,
        DEVICES) 
    torch.save(net.state_dict(), MODEL_SAVE_PATH)

def Inference():
    model, vocab = load_finetune_model(MODEL_SAVE_PATH)
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print(f'FP32 Model Size: ',end='')
    print_size_of_model(model)
    print(f'INT8 Model Size: ',end='')
    print_size_of_model(quantized_model)
    print("Loading Test Dataset...")
    testDataset = YelpDataset(DATASET_PATH,MAX_LEN,vocab,False,TRAIN_TEST_RATE)
    test_iter = torch.utils.data.DataLoader(testDataset, BATCH_SIZE)
    print('testing...')
    test_acc = d2l.evaluate_accuracy_gpu(model, test_iter)
    print(f'original test acc {test_acc:.3f}')
    test_acc = d2l.evaluate_accuracy_gpu(quantized_model, test_iter)
    print(f'quantization test acc {test_acc:.3f}')

if __name__ == "__main__":
    print('Finetuning...')
    Finetune()
    print('\n\nInferencing...')
    Inference()
