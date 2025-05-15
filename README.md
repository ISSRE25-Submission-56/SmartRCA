# AetherLog: Log-based Root Cause Analysis by Integrating Large Language Models with Knowledge Graphs

This repository contains the official implementation of our paper:  
**# AetherLog: Log-based Root Cause Analysis by Integrating Large Language Models with Knowledge Graphs**

## 🔧 Features
- Entity normalization via BigLog embedding + clustering
- Context-aware entity recall using LLM-generated summaries
- LLM-based RCA with entity-augmented prompting

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
