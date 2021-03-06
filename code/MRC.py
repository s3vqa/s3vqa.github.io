import json
import string
import re
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize 
from collections import OrderedDict
from nltk.stem import PorterStemmer
import transformers
from transformers import pipeline
import torch
transformers.logging.set_verbosity_error()
ps = PorterStemmer()
span = re.compile('\n')

qa_pipeline = pipeline(
    "question-answering",
    model="./SpanBERT/squad_output",
    tokenizer="./SpanBERT/squad_output"
)

input_question_file = #input file with questions
context_file = #input file with context
output_file = #output file with context

questions = json.load(open(input_question_file))
contexts = json.load(open(context_file))[0]

info = {}
for q_id, item in tqdm(questions.items()):
    query = item['predicted_q']
    try:
        passage = ''
        cx_g = ctxt[q_id][:]
        for j in cx_g:
            passage += ' '.join(word_tokenize(j['context'] + ' '))
        passage = ''.join([i if ord(i) < 128 else ' ' for i in passage])
        answer = qa_pipeline({"question": query, "context": passage})
        info[q_id] = [answer]
    except:
        pass
    
with open(output_file, 'w') as h:
    json.dump(info, h)
