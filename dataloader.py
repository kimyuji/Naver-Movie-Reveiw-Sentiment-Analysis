#%%
import os
import numpy as np
import pandas as pd
from torchtext.legacy import data # batchify할때 이용
import torch
# from transformers import PreTrainedTokenizer
#import konlpy
#from konlpy.tag import Okt ; okt=Okt() # 한국어 자연어 처리기
import re
import nltk # 많은 문장을 처리하기 위한 토큰화                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     모듈  # PunktSentenceTokenizer(Punkt) class
from nltk.tokenize import word_tokenize # 단어 단위 토큰화
# load data
#train_data = pd.read_csv(path+'\\Kaggle\\1_NSMC Sentiment Analysis\\ratings_train.txt', sep='\t', quoting=3)
#test_data = pd.read_csv(path+'\\Kaggle\\1_NSMC Sentiment Analysis\\ratings_test.txt', sep='\t', quoting=3)
# 불용어 사전
#stopword_100 = pd.read_csv(path+'\\Kaggle\\1_NSMC Sentiment Analysis\\한국어불용어100.txt',sep='\t',header=None)

#%%
# # 전처리 과정 함수화 
# def preprocessing(data,stopword):
#   # 특수문자 제거
#   rm = re.compile('[:;\'\"\[\]\(\)\.,@]')
#   rm_data = data.astype(str).apply(lambda x: re.sub(rm, '', x))
#   # 토큰화
#   word_token = [word_tokenize(x) for x in rm_data]
#   remove_stopwords_tokens = []
#   # 불용어 제거
#   for sentence in word_token:
#       temp = []
#       for word in sentence:
#           if word not in stopword:
#               temp.append(word)
#       remove_stopwords_tokens.append(temp)
#   return remove_stopwords_tokens
#%%
"""## 2. Batchify"""
class CustomLoader(object):
    def __init__(
        self, root, 
        train_path, test_path, predict_path,
        batch_size=64,
        valid_ratio=.2,
        max_vocab=999999,
        min_freq=1,
        use_eos=False,
        shuffle=True,
        rm = re.compile('[:;\'\"\[\]\(\)\.,@]') #제거할 특수문자
    ):
        super().__init__()
        # 전처리는 여기서 진행한다. 
        # Data Field 정의
        self.id = data.Field( # 학습에 쓰지 않을 column
            sequential=False, 
            use_vocab=False,
            unk_token=None
        )
        self.text = data.Field( 
            use_vocab=True,
            tokenize=word_tokenize,
            batch_first=True,
            include_lengths=False,
            eos_token='<EOS>' if use_eos else None
        )
        self.label = data.Field(
            sequential=False, # 0 or 1
            use_vocab=False,
            unk_token=None,
            is_target=True
        )
        
        # 데이터 읽어오기
        # ratings_train.txt : train+valid
        train, valid = data.TabularDataset(
            path = root + train_path,
            format ='tsv',
            fields = [
                ('id', self.id),
                ('text', self.text),
                ('label', self.label)],
            skip_header=True
        ).split(split_ratio=(1 - valid_ratio))

        # ratings_test.txt : test
        test = data.TabularDataset(
            path = root + test_path,
            format='tsv',
            fields=[
                ('id', self.id),
                ('text', self.text),
                ('label', self.label)],
            skip_header=True
        )

        # ko_data.csv : Kaggle commit
        predict = data.TabularDataset(
            path = root + predict_path,
            format='csv',
            fields=[
                ('id', self.id),
                ('text', self.text)],
            skip_header=True
        )

        # Batchify (Dataloader에 올리기)
        # train+valid loader
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            shuffle=shuffle,
            sort_key=lambda x: len(x.text), # 길이로 sort 후 batch 나눔!
            sort_within_batch=True, # 미니 배치 내에서 sort
        )

        # test_loader
        self.test_loader = data.BucketIterator(
            test,
            batch_size=batch_size,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            shuffle=False,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False,
        )

        # predict_loader
        self.predict_loader = data.BucketIterator(
            predict,
            batch_size=batch_size,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            shuffle=False
        )

        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq) # vocabulary set build
#%%
# loaders = CustomLoader(
#     train_path=os.getcwd()+'\\ratings_train.txt',
#     batch_size=256,
#     valid_ratio=.2,
#     device=1,
#     max_vocab=999999,
#     min_freq=5,
# )
#%%
# print("|train|=%d" % len(loaders.train_loader.dataset))
# print("|valid|=%d" % len(loaders.valid_loader.dataset))

# print("|vocab|=%d" % len(loaders.text.vocab))
# print("|label|=%d" % len(loaders.label.vocab))

# %%
