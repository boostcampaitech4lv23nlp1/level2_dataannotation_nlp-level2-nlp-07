from transformers import ElectraTokenizer,ElectraConfig,ElectraModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer,EarlyStoppingCallback, AutoModel
from torch import nn
import torch
from collections import OrderedDict
import numpy as np
class REModel(nn.Module):
    def __init__(
        self,
        model_config = None,
        pretrained_id: str = "klue/roberta-large",
        num_labels: int = 15,
        dropout_rate: float = 0.1,
        device = 'cuda:0'
    ):
        super(REModel, self).__init__()
        self.num_labels = num_labels
        if model_config is None:
            model_config = AutoConfig.from_pretrained(pretrained_id)

        self.hidden_size = model_config.hidden_size
        self.device = device
        self.tanh = nn.Tanh().to(self.device)
        if pretrained_id:
            self.plm = AutoModel.from_pretrained(pretrained_id,add_pooling_layer = False).to(device) # 5e-5
            self.dense_for_cls = nn.Sequential(OrderedDict({'cls_Linear' : nn.Linear(self.hidden_size , self.hidden_size)})).to(device)
            self.dense_for_e1 =  nn.Sequential(OrderedDict({'e1_Linear' : nn.Linear(self.hidden_size , self.hidden_size)})).to(device)
            self.dense_for_e2 = nn.Sequential(OrderedDict({'e2_Linear' : nn.Linear(self.hidden_size , self.hidden_size)})).to(device)
            self.entity_classifier = nn.Sequential(OrderedDict({                          # 10e-5
                                    'dense1': nn.Linear(self.hidden_size * 3, self.hidden_size),
                                    'dropout1': nn.Dropout(dropout_rate),
                                    'dense2': nn.Linear(self.hidden_size, self.hidden_size),
                                    # 'dropout2': nn.Dropout(dropout_rate),
                                    'out_proj': nn.Linear(self.hidden_size, num_labels)
                                    })).to(device)
        self.num_labels = num_labels

    def forward(self, inputs):
        # print(inputs)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        entity_idx= inputs['Entity_idxes']
        if input_ids.is_cuda ==False:
            print('merong',input_ids.dtype)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
        x = self.plm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state
        
        subject_entity_avgs, object_entity_avgs = self.get_avg_entity(x,entity_idx)
        subject_entity_avgs = torch.as_tensor(subject_entity_avgs, dtype=torch.float, device=self.device)
        object_entity_avgs = torch.as_tensor(object_entity_avgs, dtype=torch.float, device=self.device)
        x0 = x[:, 0, :]                                         # cls 16, hidden
        # print('x0 type',x0.device, 'subject_entity_avgs', subject_entity_avgs.device, 'object_entity_avgs', object_entity_avgs.device)
        x0 = self.dense_for_cls(self.tanh(x0))# 배치, 히든
        subject_entity_avgs = self.dense_for_e1(self.tanh(subject_entity_avgs)) # 배치, 히ㄷ,ㄴ
        object_entity_avgs = self.dense_for_e2(self.tanh(object_entity_avgs))    # 배피, 히든

        x = torch.cat([x0 ,subject_entity_avgs, object_entity_avgs], dim=1) # 배치 , 히든 *3
        x = self.entity_classifier(x) # 배치, 넘라벨:30
        return {'output' :x} # 배치, 넘라벨

    def get_avg_entity(self,output,entity_idx):
        subject_entity_avgs = []
        object_entity_avgs = []
        for idx in range(entity_idx.shape[0]):
            subject_entity_avg = output[idx, entity_idx[idx][0]:entity_idx[idx][1], :] # 1,n,hidden -> n,hidden -> 1,hidden

            subject_entity_avg = torch.mean(subject_entity_avg, dim=0) # 1,hidden -> 1,hidden

            subject_entity_avgs.append(subject_entity_avg.cpu().detach().numpy())

            object_entity_avg = output[idx, entity_idx[idx][2]:entity_idx[idx][3], :] # 1, len, hidden

            object_entity_avg = torch.mean(object_entity_avg, dim=0)           #  1, hidden
            
            object_entity_avgs.append(object_entity_avg.cpu().detach().numpy()) # 1, 1, hidden

        subject_entity_avgs = torch.tensor(np.array(subject_entity_avgs)) # batch,1,hidden
        object_entity_avgs = torch.tensor(np.array(object_entity_avgs))
        return subject_entity_avgs, object_entity_avgs