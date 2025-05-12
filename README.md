# SmartRCA: Bridging Structured Knowledge Graph and Large Language Model for Explainable Fault Root Cause Analysis in Logs

This repository contains the official implementation of our paper:  
**# SmartRCA: Bridging Structured Knowledge Graph and Large Language Model for Explainable Fault Root Cause Analysis in Logs**

## 🔧 Features
- Entity normalization via BERT embedding + clustering
- Context-aware entity recall using LLM-generated summaries
- LLM-based classification with entity-augmented prompting

## 📁 Project Structure
- `scripts/`: core pipeline modules
- `models/`: prompt design & LLM inference wrappers
- `utils/`: helper functions
- `data/`: preprocessed logs for CMCC and ZTE
- `results/`: output files and figures

## 🔨 Installation
```bash
conda create -n smartrca python=3.10
conda activate smartrca
pip install -r requirements.txt
