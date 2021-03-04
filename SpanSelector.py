import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel
from transformers import AutoModel  

class SpanSelector(nn.Module):
    def __init__(self, num_labels=2, hidden_size=768, temperature = torch.tensor([0.0])):
        super(SpanSelector, self).__init__()
        self.num_labels = num_labels
        self.device = None
        self.embedding_size = hidden_size
        self.bert = BertModel.from_pretrained('bert-base-uncased') #AutoModel.from_pretrained("SpanBERT/spanbert-base-cased")
        self.fc = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, data):
        outputs = self.bert(input_ids=data['question_input_ids'].cuda())

        question_bert_output = outputs[0] 
        logits = self.fc(question_bert_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        label_i = data['label_i'].cuda()
        label_j = data['label_j'].cuda()

        loss = (self.ce(start_logits,label_i) + self.ce(end_logits,label_j)) / 2.0

        i = torch.max(start_logits,dim=1)[1]
        j = torch.max(end_logits,dim=1)[1]

        accuracy = torch.sum((i == label_i) * (j == label_j))

        return loss, (accuracy,torch.tensor([0.0]),torch.tensor([0.0])), (i, j)