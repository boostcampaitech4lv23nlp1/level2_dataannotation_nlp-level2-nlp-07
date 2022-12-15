## in load_data.py
import pickle as pickle
import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from  utils import *
class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, dataset,labels,tokenizer,cfg):
        self.labels = labels
        self.tokenizer = tokenizer
        self.new_tokens = []
        self.marker_mode = cfg.train.marker_mode        
        self.dataset = self.tokenizing(dataset)

        self.cfg = cfg
    def __getitem__(self, idx):
        if self.cfg.train.entity_embedding:
            if len(self.labels) ==0:
                return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                        'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                        'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
                        'Entity_idxes': torch.LongTensor(self.dataset[idx]['Entity_idxes']).squeeze(0)                    
                            }
            else:
                return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                        'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                        'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
                        # 'Entity_type_embedding': torch.LongTensor(self.dataset[idx]['Entity_type_embedding']).squeeze(0),
                        'Entity_idxes': torch.LongTensor(self.dataset[idx]['Entity_idxes']).squeeze(0),
                        'labels' : torch.LongTensor([self.labels[idx]]).squeeze()}
        else:
            if len(self.labels) ==0:
                return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                        'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                        'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0)                    
                            }
            else:
                return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                        'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                        'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
                        'labels' : torch.LongTensor([self.labels[idx]]).squeeze()}
    def __len__(self):
        return len(self.dataset)
    
    def tokenizing(self,dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text= self.add_special_enti(item,marker_mode = self.marker_mode)
            # text = '[SEP]'.join([concat_entity, item['sentence']])
            outputs = self.tokenizer(text, add_special_tokens=True,
                                          truncation=True,
                                          return_tensors="pt",
                                        #   padding='max_length',
                                        #   max_length=256
                                    )
            sent = outputs['input_ids'].squeeze(0).numpy().tolist()
            Entity_type_embedding, Entity_idxes = self.get_embed_idx(sent)
            outputs['Entity_type_embedding'] = Entity_type_embedding
            outputs['Entity_idxes'] = Entity_idxes
            data.append(outputs)
        return data
    
    def add_special_enti(self,df,marker_mode= None):
        def change_enti(sub,obj,sub_type, obj_type,marker_mode = None):
            if marker_mode == 'TEM_punct':
                Eng_type_to_Kor = {"STU":"학문", "LAN":"언어", "PER" : "사람", "ORG" : "기관", "DAT" : "날짜", "LOC" : "장소", "POH" : '기타'}                
                marked_sub = ['@']+['*']+list(Eng_type_to_Kor[sub_type]) + ['*']+list(sub)+['@']
                marked_obj = ['#']+['^']+list(Eng_type_to_Kor[obj_type]) + ['^']+list(obj)+['#']
            return marked_sub, marked_obj
        marked = []
        # print(type(df['subject_entity']),df['subject_entity'] )
        sub = df['subject_entity'].split(',')[2].split(':')[1]
        s_s = int(df['subject_entity'].split(',')[0].split(':')[1])
        s_e = int(df['subject_entity'].split(',')[1].split(':')[1]) + 1
        sub_type = df['subject_entity'].split(',')[3].split(':')[1].lstrip()[:-1]
        obj = df['object_entity'].split(',')[2].split(':')[1]
        o_s = int(df['object_entity'].split(',')[0].split(':')[1])
        o_e = int(df['object_entity'].split(',')[1].split(':')[1]) + 1
        obj_type = df['subject_entity'].split(',')[3].split(':')[1].lstrip()[:-1]

        marked_sub,marked_obj = change_enti(sub,obj,sub_type,obj_type,marker_mode = marker_mode)
        if s_s < o_s:
            marked += df['sentence'][:s_s]
            marked += marked_sub
            marked += df['sentence'][s_e:o_s]
            marked += marked_obj
            marked += df['sentence'][o_e:]
            marked = ''.join(marked)
        else:
            marked += df['sentence'][:o_s]
            marked += marked_obj
            marked += df['sentence'][o_e:s_s]
            marked += marked_sub
            marked += df['sentence'][s_e:]
            marked = ''.join(marked)
        return marked

    def get_embed_idx(self,sent):
        subj_1 = self.tokenizer.convert_tokens_to_ids('@')
        subj_2 = self.tokenizer.convert_tokens_to_ids('*')
        obj_1 = self.tokenizer.convert_tokens_to_ids('#')
        obj_2 = self.tokenizer.convert_tokens_to_ids('^')
        names = self.tokenizer.convert_tokens_to_ids(["학문","언어","사람", "기관", "날짜","장소",'기타'])
        embed = [0 for _ in range(len(sent))]
        sub_s,sub_e,obj_s,obj_e = 0,0,0,0
        
        for idx,t in enumerate(sent):
            if t == subj_1 and sent[idx+1] == subj_2 and (sent[idx+2] in names):
                sub_s = idx + 4
                sub_e = sub_s + 1
                while sent[sub_e] != subj_1:
                    sub_e += 1
                break
        for idx,t in enumerate(sent):
            if t == obj_1 and sent[idx+1] == obj_2 and (sent[idx+2] in names):
                obj_s = idx + 4
                obj_e = obj_s + 1
                while sent[obj_e] != obj_1:
                    obj_e += 1
                break

        embed[sub_s:sub_e+1] = [1 for _ in range(sub_s,sub_e+1)]
        embed[obj_s:obj_e+1] = [1 for _ in range(obj_s,obj_e+1)]
        return embed,np.array([sub_s,sub_e,obj_s,obj_e])


    
        
def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    return pd_dataset


class CoRE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, dataset,labels,tokenizer,cfg,mode='mask1'):
        self.labels = labels
        self.tokenizer = tokenizer
        self.mode = mode
        dataset, label_constraint= self.tokenizing(dataset)
        self.dataset  = dataset
        self.label_constraint = label_constraint
        self.cfg = cfg
    def __getitem__(self, idx):
        if len(self.labels) ==0:
            return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                    'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                    'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
                    'Entity_idxes': torch.LongTensor(self.dataset[idx]['Entity_idxes']).squeeze(0),
                    'label_constraints' : torch.Long
                        }
        else:
            return {'input_ids': torch.LongTensor(self.dataset[idx]['input_ids']).squeeze(0),
                    'attention_mask': torch.LongTensor(self.dataset[idx]['attention_mask']).squeeze(0),
                    'token_type_ids': torch.LongTensor(self.dataset[idx]['token_type_ids']).squeeze(0),
                    'Entity_idxes': torch.LongTensor(self.dataset[idx]['Entity_idxes']).squeeze(0),
                    'labels' : torch.LongTensor([self.labels[idx]]).squeeze()}

    def __len__(self):
        return len(self.dataset)
    
    def tokenizing(self,dataframe):
        data = []
        label_constraint = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            sub_type = eval(item['subject_entity'])['type']
            obj_type = eval(item['object_entity'])['type']
            tmp = self.get_label_constraint(sub_type,obj_type)
            text= self.extract_word(item,self.mode)
            # text = '[SEP]'.join([concat_entity, item['sentence']])
            outputs = self.tokenizer(text, add_special_tokens=True,
                                          truncation=True,
                                          return_tensors="pt",
                                          padding='max_length',
                                          max_length=16
                                    )
            sent = outputs['input_ids'].squeeze(0).numpy().tolist()
            Entity_idxes = self.get_embed_idx(sent)
            outputs['Entity_idxes'] = Entity_idxes
            data.append(outputs)
            label_constraint.append(tmp)
        # self.label_constraint = np.array(label_constraint)
        return data,np.array(label_constraint)
    def extract_word(self,df,mode):
        if mode == 'mask1':
            sub = eval(df['subject_entity'])['text']
            obj = eval(df['object_entity'])['text']

            # return sub+':'+ obj,label_constraint
            return sub+':'+ obj
        else:
            return ' '
    def get_embed_idx(self,sent):
        sep = self.tokenizer.convert_tokens_to_ids(':')
        if sent[0]==0 and sent[1]==2:
            return [0,1,0,1]
        i=1
        s_s,s_e,o_s,o_e = 1,0,0,0
        while sent[i] != 2:
            i+=1
            if sent[i] == sep:
                s_e = i
                o_s = i+1
        o_e = i
        return np.array([s_s,s_e,o_s,o_e])

    def get_label_constraint(self,sub_type,obj_type):
        '''
        input  : sub_type,obj_type
        output : 30개의 원소를 갖는 리스트
                각 원소는 sub_type,obj_type : 
        '''
        sol = pd.read_csv('./data/solsol.csv')
        base = np.zeros(30)
        # print(type(sol[(sol['subj_type']==sub_type)&(sol['obj_type']==obj_type)]['label'].iloc[0]))
        try:
            const = sorted(label_to_num(eval(sol[(sol['subj_type']==sub_type)&(sol['obj_type']==obj_type)]['label'].iloc[0])))
            base[const] = 2
        except:
            base[0] = 2
        return base