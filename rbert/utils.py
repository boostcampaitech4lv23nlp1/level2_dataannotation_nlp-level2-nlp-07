from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import sklearn
import numpy as np
import pickle as pickle
from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
# from torchsampler import ImbalancedDatasetSampler
from transformers import DataCollatorWithPadding


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation','stu:sub_study','stu:high_study','stu:alternate_names','stu:contributor',
            'stu:area','stu:research_group','stu:influence','stu:element','lan:high_language',
            'lan:sub_language','lan:product','lan:use_area','lan:alternate_names','lan:group_of_people']
    # no_relation class를 제외한 micro F1 score
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(14)[labels]

    score = np.zeros((14,))
    for c in range(14):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    # auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    return {
      'micro_f1_score': f1,
    #   'auprc' : auprc,
      'accuracy': acc,
    }

def label_to_num(label):
    num_label = []
    with open('/opt/ml/code/rbert/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f) # dict
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss()
        ce_loss = ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class TrainerwithFocalLoss(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")
        # inputs = {k : v.to(device) for k,v in inputs.items()}
        # forward pass
        outputs = model(inputs)
        # logits = outputs.get("logits")
        logits = outputs['output']


        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = FocalLoss()
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss