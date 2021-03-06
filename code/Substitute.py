import torch
from transformers import BertTokenizer
import re
from tqdm import tqdm
import json
import logging
logging.basicConfig(level=logging.ERROR)
max_sequence_length_question = 32


def reformulate(input_file):
#     dataset = json.load(open(input_file))
    dataset = input_file
    count=0
    f_count = 0
    for q_id, value in dataset.items(): 
        try:
            question = value['question']
            label_hyper = value['P_hypernym']
            obj = sorted(value['hyponyms_scores_l'], key=value['hyponyms_scores_l'].get, reverse=True)[0] 
            span = re.compile(label_hyper, re.IGNORECASE)
            hypo = obj
            if ' ' == label_hyper[0]:
                hypo = ' ' + hypo
            if ' ' == label_hyper[-1]:
                hypo = hypo + ' '
            new_question = span.sub(hypo, question)
            value['predicted_q'] = new_question
        except:
            value['predicted_q'] = value['question']
            f_count+=1
#     print(f_count)
    return dataset

def Substitute(input_file, ckpt_path_H):
#     dataset = json.load(open(input_file))
    dataset = input_file
    l = len(dataset)
    print("Dataset Size : ", len(dataset))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_labels = 2
    hidden_size = 768
    model = HypoSelector(num_labels, hidden_size)
    model_save_filepath = # checkpoint for HypoSelector
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
            hyponyms = d["detections"]
            hyponym_input_ids = []

            for hypo in hyponyms:
                hypo_encoded = tokenizer.encode(hypo,max_length=8,pad_to_max_length=True,truncation=True)
                hyponym_input_ids.append(hypo_encoded)
            while(len(hyponym_input_ids) < 32):
                hyponym_input_ids.append(torch.zeros(8, dtype=torch.int64))
            hyponym_input_ids = torch.tensor(hyponym_input_ids, dtype=torch.int64).unsqueeze(0)

            label_hyper_encoded = tokenizer.encode(d["P_hypernym"],pad_to_max_length=False,add_special_tokens=False)	
            def find_sub_list(sl,l):
                sll=len(sl)
                for ind in (i for i,e in enumerate(l) if e==sl[0]):
                    if l[ind:ind+sll]==sl:
                        return ind,ind+sll-1
            label_i,label_j = 0,0
            try:
                label_i,label_j = find_sub_list(label_hyper_encoded, question_encoded)
            except:
                continue

            question = d["question"]
            label_hyper = d["P_hypernym"]
            label_hypo = '+'
            detected_hyponyms = d["detections"]
            o2hs = d["P_o2hs"]
            hypo_scores = []
            for detected_hypo in detected_hyponyms:
                hypo_scores.append(o2hs[detected_hypo])
            while(len(hypo_scores) < 32):
                hypo_scores.append(0.0)
            new_questions = []
            new_is, new_js = [], []
            span = re.compile(label_hyper, re.IGNORECASE)
            hypo = label_hypo
            if ' ' == label_hyper[0]:
                hypo = ' ' + label_hypo
            if ' ' == label_hyper[-1]:
                hypo = label_hypo + ' '
            new_question = span.sub(hypo, question)
            new_question_encoded = tokenizer.encode(new_question,
                                                         max_length=max_sequence_length_question,
                                                         pad_to_max_length=True,truncation=True)
            hypo_encoded = tokenizer.encode(hypo, pad_to_max_length=False,
                                                        add_special_tokens=False)   

            def find_sub_list(sl,l):
                sll=len(sl)
                for ind in (i for i,e in enumerate(l) if e==sl[0]):
                    if l[ind:ind+sll]==sl:
                        return ind,ind+sll-1
            try:
                new_i, new_j = find_sub_list(hypo_encoded, new_question_encoded)
            except:
                continue

            flag = 0
            for detected_hypo in detected_hyponyms:
                span = re.compile(label_hyper, re.IGNORECASE)
                hypo = detected_hypo
                if ' ' == label_hyper[0]:
                    hypo = ' ' + detected_hypo
                if ' ' == label_hyper[-1]:
                    hypo = detected_hypo + ' '
                new_question = span.sub(hypo, question)
                new_question_encoded = tokenizer.encode(new_question,
                                                             max_length=max_sequence_length_question,
                                                             pad_to_max_length=True,truncation=True)
                hypo_encoded = tokenizer.encode(hypo, pad_to_max_length=False,
                                                            add_special_tokens=False)   
                def find_sub_list(sl,l):
                    sll=len(sl)
                    for ind in (i for i,e in enumerate(l) if e==sl[0]):
                        if l[ind:ind+sll]==sl:
                            return ind,ind+sll-1
                try:
                    new_i, new_j = find_sub_list(hypo_encoded, new_question_encoded)
                    flag = 1
                except:
                    continue
                new_questions.append(new_question_encoded)
                new_is.append(new_i)
                new_js.append(new_j)

            if flag!=1:
                continue

            while(len(new_questions)<32):
                new_is.append(0.0)
                new_js.append(0.0)
                new_questions.append([0]*max_sequence_length_question)

            new_qs = torch.tensor(new_questions, dtype=torch.int64).unsqueeze(0)
            new_is = torch.tensor(new_is, dtype=torch.int64).unsqueeze(0)
            new_js = torch.tensor(new_js, dtype=torch.int64).unsqueeze(0)

            hypo_scores = torch.tensor(hypo_scores, dtype=torch.int64).unsqueeze(0)

            _,_,(hypo_i, scores, logit) = model({"question_input_ids" : question_encoded_tensor, 
                                                "detected_hyponym_input_ids" : hyponym_input_ids, 
                                                "label_hypo" : torch.tensor([0], dtype=torch.int64), 
                                                "label_i" : torch.tensor([label_i], dtype=torch.int64), 
                                                "label_j" : torch.tensor([label_j], dtype=torch.int64),
                                                'new_gquestion' : torch.tensor([0]*32, dtype=torch.int64),
                                                'new_gi' : torch.tensor([0], dtype=torch.int64),
                                                'new_gj' : torch.tensor([0], dtype=torch.int64),
                                                'new_questions' : new_qs,
                                                'new_is' : new_is,
                                                'new_js' : new_js,
                                                'hypo_scores' : hypo_scores,
                                                })

            hypo_i = hypo_i.item()
            scores = scores[0]
            scores_l = logit[0]
            hyponyms_scores = {}
            hyponyms_scores_l = {}
            i = 0

            for h in hyponyms:
                hyponyms_scores[h] = scores[i].item()
                i += 1
            i=0
            for h in hyponyms:
                hyponyms_scores_l[h] = scores_l[i].item()
                i += 1
            try:
                d['P_hyponym'] = hyponyms[hypo_i]
                d['hyponyms_scores'] = hyponyms_scores
                d['hyponyms_scores_l'] = hyponyms_scores_l
                d['hypernymy_relations'] = {}
            except:
                hypo = sorted(hyponyms_scores_l, key=hyponyms_scores_l.get, reverse=False)[0]
                d['P_hyponym'] = hypo
                d['hyponyms_scores'] = hyponyms_scores
                d['hyponyms_scores_l'] = hyponyms_scores_l
                d['hypernymy_relations'] = {}
                f_count+=1
    output_file = reformulate(dataset)
    with open('substitute_output.json', 'w') as h:
        json.dump(output_file, h)
