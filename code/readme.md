# Query reformulation for S3VQA
This repository contains the code for : Select, Substitute, Search - New Benchmark for Knowledge-Augmented Visual Question Answering

## Installation 

### Requirement 
Python 3.6  
Pytorch 1.4  

```bash
conda create -n qr_vqa python=3.6
conda activate qr_vqa 
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

## Files

### models/HypoSelector
To extract the correct hyponym from the detections needed for replacement.

### models/SpanSelector
To extract the span/hypernym from the question that needs to be replaced.

### preprocessing/data
for preprocessing of data

### Select, Substitute, and Search 

 - Select : to predict relevant hypernym and hypernym-to-hyponym scores
 - Substitute : to predict relevant hyponym and reformulate the question
 - google_search, MRC : to retrieve external knowledge and predict natural language answer
#### Note
 Select quantifies hypernym-hyponym relation making use of Poincaré Embeddings. Please refer to this https://github.com/facebookresearch/poincare-embeddings directory for setup.

### demo/demo_pipeline
jupyter notebook with a sample example on Select, Substitute and Search

## Training 
```bash
python run.py
```



!!We are working on more updates!!
