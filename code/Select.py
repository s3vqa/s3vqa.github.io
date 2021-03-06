import torch
from transformers import BertTokenizer
from tqdm import tqdm
import json
import logging
logging.basicConfig(level=logging.ERROR)
from hype.hypernymy_eval import EntailmentConeModel
import argparse
import os
import torch as th
from nltk.corpus import wordnet
import nltk
import numpy as np
from tqdm import tqdm 
import json
import spacy
nlp = spacy.load("en_core_web_sm")

def find_max(detection, hyper, p_model):
    with th.no_grad():
        dists = p_model.predict_many(detection[1], [hyper]*len(detection[1]))
        dists = [np.exp(score)*100 for score in dists]
        return (detection[0], max(dists))
    
def m_hyper(hyper):
    doc = nlp(hyper)
    x = {}
    for token in doc:
        x.setdefault(token.pos_, []).append(token.text)
        if 'NOUN' in x:
            return '_'.join(x['NOUN'])
        else:
            return hyper

def hypo_hyper_scores(input_file):
#     dataset = json.load(open(input_file))
    dataset = input_file
    count, fcount = 0, 0
    chkpnt = # checkpoint for Poincare model
    if isinstance(chkpnt, str):
        assert os.path.exists(chkpnt)
        chkpnt = th.load(chkpnt, map_location='cuda:0')
    p_model = EntailmentConeModel(chkpnt)
    for q_id, item in tqdm(dataset.items()):
        try:
            o_hyper = item['P_hypernym']
            detections = item['detections']
            hyper = m_hyper(o_hyper)
            hyper = wordnet.synsets(hyper)[0].name() if wordnet.synsets(hyper) else hyper
            detections = ['_'.join(i.split()) for i in detections]
            detections = [(detection, [i.name() for i in wordnet.synsets(detection)]) if wordnet.synsets(detection) else (detection, [detection]) for detection in detections]    
            o2hs = {}
            with th.no_grad():
                result = []
                for detection in detections:
                    result.append(find_max(detection, hyper, p_model))
                result = sorted(result, key=lambda x:-x[1])
                for p, q in result:
                    o2hs[' '.join(p.split('_'))] = q
            item['P_o2hs'] = o2hs
            count+=1
        except:
            item['P_o2hs'] = {}
            fcount+=1
#     print('\n', count, fcount)
    return dataset

def Select(input_file, ckpt_path_S):
#     dataset = json.load(open(input_file))
    dataset = input_file
    l = len(dataset)
    print("Dataset Size : ", len(dataset))
    num_labels = 2
    hidden_size = 768
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SpanSelector(num_labels, hidden_size)
    model_save_filepath = #checkpoint SpanSelector
    state = torch.load(model_save_filepath)
    model.load_state_dict(state['state_dict'])
    device = torch.device('cuda')
    print("Using the device : ", device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for q_id, d in tqdm(dataset.items()):
            question = d["question"]
            question_encoded = tokenizer.encode(question,max_length=32,pad_to_max_length=True,truncation=True)
            question_encoded_tensor = torch.tensor(question_encoded, dtype=torch.int64).unsqueeze(0)

            _,_,(i,j) = model({"question_input_ids" : question_encoded_tensor, 
                                       "label_i" : torch.tensor([0], dtype=torch.int64), 
                                       "label_j" : torch.tensor([0], dtype=torch.int64)})
            i = i.item()
            j = j.item()
            span = tokenizer.decode(question_encoded_tensor[0][i:j+1])
            d['P_hypernym'] = span
    output_file = hypo_hyper_scores(dataset)
    with open('select_output.json', 'w') as h:
        json.dump(output_file, h)
