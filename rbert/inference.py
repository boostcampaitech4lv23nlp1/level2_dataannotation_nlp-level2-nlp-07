from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F
from utils import *
import pickle as pickle
import numpy as np
from tqdm import tqdm
from model import *
from transformers import DataCollatorWithPadding


def inference(model, tokenized_sent, device,tokenizer):
    """
    test dataset을 DataLoader로 만들어 준 후, batch_size로 나눠 model이 예측 합니다.
    """
    collate_fn = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False ,collate_fn = collate_fn) # batch_size= 16
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        # print('batch',data['input_ids'].device)
        # print('batch',data['input_ids'].shape)
        data = {k:v.to(device) for k,v in data.items()}
        # print('batch in device',data['input_ids'].device)
        with torch.no_grad():
            outputs = model(data) # default : input, token  다 넣어줬음 원래
            logits = outputs['output']
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            result = np.argmax(logits, axis=-1)

            output_pred.append(result)
            output_prob.append(prob)
  
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(cfg, label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open(cfg.test.num_to_label, 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
  
    return origin_label

def load_test_dataset(dataset_dir):
    """
    test dataset을 불러온 후, tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))

    return test_dataset['id'], test_dataset, test_label

def test(cfg):
    ## Device
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    ## load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(cfg.model.saved_model)
    model = REModel()
    model.load_state_dict(torch.load(cfg.model.saved_model))
    # model = torch.load(cfg.model.saved_model).to(device)
    # print(model)
    model.parameters
    # model.to(device)
    # print(model.get_device())

    ## load test datset
    test_dataset_dir = cfg.data.test_data
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label, tokenizer,cfg)
    
    ## predict answer ## 절대 바꾸지 말 것 ##
    pred_answer, output_prob = inference(model, Re_test_dataset, device,tokenizer) # model에서 class 추론
    pred_answer = num_to_label(cfg, pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
    # gold_answer = num_to_label(cfg,test_label)

    # print(compute_metrics(gold_answer,pred_answer))
    ## make csv file with predicted answer
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

    output.to_csv(cfg.test.output_csv, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.

def compute_metrics(labels,preds):
    """ validation을 위한 metrics function """
    # labels = label
    # preds = pred.predictions.argmax(-1)
    # probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    # auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
      'micro_f1_score': f1,
    #   'auprc' : auprc,
      'accuracy': acc,
    }