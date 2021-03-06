import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
 

class HypoSelector(nn.Module):
    def __init__(self, num_labels=2, hidden_size=768, temperature = torch.tensor([0.0])):
        super(HypoSelector, self).__init__()
        self.num_labels = num_labels
        self.device = None
        self.embedding_size = hidden_size
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding = self.bert.get_input_embeddings()
        self.fc = nn.Linear(hidden_size,2048)
        self.fc1 = nn.Linear(2048,1024)
        self.dropout = nn.Dropout(p=0.6)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.relu = nn.Tanh()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.eps = 1e-9
        self.margin = .6
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.fcb1 = nn.Linear(2048, 1024)
        self.fcb2 = nn.Linear(1024, 512)
        self.fcb3 = nn.Linear(512, 64)
        self.fcb4 = nn.Linear(64, 1)
        self.fcb = nn.Linear(2, 1)
        self.fcl = nn.Linear(2, 1, bias=False)
        self.p = nn.Parameter(torch.tensor(0.5))
        self.alpha = torch.tensor(0.5)
        self.alpha.requires_grad = True
 
    def forward(self, data):
        B,S = data['question_input_ids'].shape
        _,N,SH = data['detected_hyponym_input_ids'].shape
        i = data["label_i"].cuda()
        j = data["label_j"].cuda()
 
        masked_question = torch.zeros(B,S, dtype=torch.int64).cuda()
        masked_index = torch.zeros(B, dtype=torch.int64).cuda()
        for b in range(B): 
            if j[b] >= i[b]:
                masked_question[b,0:S-j[b]+i[b]] = torch.cat((data['question_input_ids'][b,0:i[b]], torch.tensor([103],dtype=torch.int64), data['question_input_ids'][b,j[b]+1:S]),0)
                masked_index[b] = i[b]
        outputs = self.bert(input_ids=data['question_input_ids'].cuda())

        question_bert_output = outputs[0]  # B X S X 786
        predictions = torch.zeros(B,self.embedding_size).cuda()
        for b in range(B):
            predictions[b] = torch.sum(question_bert_output[b][i[b]:j[b]+1],dim=0) / (j[b] - i[b] + 1)
        embedded_detected_hyponym = torch.sum(self.embedding(data['detected_hyponym_input_ids'].cuda().view(B*N,SH)),dim=1)  #.view(B,N,self.embedding_size)        
        label_hypo = data['label_hypo'].cuda()
        
        pos_q = torch.zeros(B, 1, S, dtype=torch.int64).cuda()
        neg_q = torch.zeros(B, N-1, S, dtype=torch.int64).cuda()
        
        pos_i = torch.zeros(B, 1, 1, dtype=torch.int64).cuda()
        pos_j = torch.zeros(B, 1, 1, dtype=torch.int64).cuda()
        neg_i = torch.zeros(B, N-1, 1, dtype=torch.int64).cuda()
        neg_j = torch.zeros(B, N-1, 1, dtype=torch.int64).cuda()

        nchor = torch.zeros(B, 768).cuda()
        for b in range(B): 
            ss = torch.arange(N).cuda()
            ss = ss[ss != label_hypo[b]]

            pos_q[b] = data['new_questions'][b, label_hypo[b]] 
            neg_q[b] = data['new_questions'][b, ss] 
            
            pos_i[b] = data['new_is'].view(B, N, 1)[b, label_hypo[b]] 
            pos_j[b] = data['new_js'].view(B, N, 1)[b, label_hypo[b]] 
            
            neg_i[b] = data['new_is'].view(B, N, 1)[b, ss]  
            neg_j[b] = data['new_js'].view(B, N, 1)[b, ss] 
            
            nchor[b] = predictions[b]
              
        pos_q = pos_q.reshape(B, S)
        pos_i = pos_i.reshape(B, 1)
        pos_j = pos_j.reshape(B, 1)
        ngout = self.bert(input_ids=pos_q)
        new_gold_question_bert_output = ngout[0]
        pos_element = torch.zeros(B, self.embedding_size).cuda()
        for b in range(B):
            pos_element[b] = torch.sum(new_gold_question_bert_output[b][pos_i[b]:pos_j[b]+1], dim=0)/(pos_j[b]-pos_i[b]+1)
            
        neg_q = neg_q.reshape(-1, S)
        neg_i = neg_i.reshape(-1, 1)
        neg_j = neg_j.reshape(-1, 1)
        nout = self.bert(input_ids=neg_q)
        new_questions_bert_output = nout[0]
        neg_elements = torch.zeros(B*(N-1), self.embedding_size).cuda()
        for b in range(B*(N-1)):
            neg_elements[b] = torch.sum(new_questions_bert_output[b][neg_i[b]:neg_j[b]+1], dim=0)/(neg_j[b] - neg_i[b] + 1)
            
        pos = pos_element.repeat_interleave(N-1, dim=0)
        anchor = predictions.repeat_interleave(N-1, dim=0)
        neg = neg_elements
        
        rep_pos = self.fc1(self.relu(self.fc(pos)))
        rep_nchor = self.fc1(self.relu(self.fc(anchor)))
        rep_neg = self.fc1(self.relu(self.fc(neg)))

        distances_pos = 1 - self.cos(rep_nchor, rep_pos).view(-1, 1)
        distances_neg = 1 - self.cos(rep_nchor, rep_neg).view(-1, 1)
        losses = F.relu(distances_pos + self.margin - distances_neg)
        loss = losses.mean()

        all_q = data['new_questions'].cuda().view(-1, S)
        all_i = data['new_is'].cuda().view(-1, 1)
        all_j = data['new_js'].cuda().view(-1, 1)
        
        all_out = self.bert(input_ids=all_q)
        all_questions_bert_output = all_out[0]
        all_elements = torch.zeros(B*N, self.embedding_size).cuda()
        for b in range(B*N):
            all_elements[b] = torch.sum(all_questions_bert_output[b][all_i[b]:all_j[b]+1], dim=0)/\
            (all_j[b] - all_i[b] + 1)
            
        nchor = predictions.repeat_interleave(N, dim=0)
        rep_all_elements = self.fc1(self.relu(self.fc(all_elements)))
        rep_nchor = self.fc1(self.relu(self.fc(nchor)))
        
        dists = 1 - self.cos(rep_nchor, rep_all_elements).view(-1, 1)
        o2hs = data['hypo_scores'].cuda().view(-1, 1)
        logit = self.fcb(torch.cat((dists, o2hs), 1).cuda())
        
        label = torch.zeros((B,N)).cuda()
        for b in range(B):
            label[b][label_hypo[b]] = 1
        label = label.view(-1, 1)

        loss_b = self.bce(logit, label)
        loss = .9*(losses.mean()) + .1*loss_b

        scores = 1 - self.cos(rep_nchor, rep_all_elements).view(B, N)
        hypo_i = torch.max(logit.view(B, N), dim=1)[1]

        accuracy = torch.sum(hypo_i == label_hypo)
        return loss, (accuracy,torch.tensor([0.0]),torch.tensor([0.0])), (hypo_i,scores, logit.view(B,N))