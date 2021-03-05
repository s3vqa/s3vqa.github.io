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

### HypoSelector
To extract the correct hyponym from the detections needed for replacement.

### SpanSelector
To extract the span/hypernym from the question that needs to be replaced.

### data
For preprocessing of data


## Training 
```bash
python run.py
```



We are working on more updates.
