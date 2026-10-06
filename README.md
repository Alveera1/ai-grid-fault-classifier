# AI Grid Fault Classifier

Classifying normal operation and electrical fault conditions (NF, LG, LL, LLG, LLL, LLLG) from PMU voltage, current and frequency data, using Python, Decision Tree, Random Forest and SVM.

## Folder structure
- data/raw/PMU_Fault_Data.csv : original data, never edited
- data/processed/clean.csv : cleaned data (duplicates and warm-up rows removed)
- common.py : shared feature list, labels, random_state and split function
- notebooks/ : 01_data_cleaning, 02_visualization, 03_modeling, 04_evaluation
- figures/ : saved plots
- models/ : saved models
- report/ : report and slides
- data_dictionary.md : column meanings

## How to run
1. Clone the repo and create an environment: conda create -n gridfault python=3.11
2. Activate it and run: pip install -r requirements.txt
3. Start Jupyter from the repo root folder: jupyter notebook
4. Run the notebooks in order (01 to 04). Every notebook starts with: from common import *

## Team rules
- One person, one notebook, one branch. Only edit your own files.
- common.py is changed only by Member 1.
- Split data by time blocks (blocked_split), never by random rows.
- Never use time_stamp as a feature. random_state = 42 everywhere.
- Every change goes through a Pull Request reviewed by another member.
- Review chain: 1 reviews 2, 2 reviews 3, 3 reviews 4, 4 reviews 1.
- Run git pull before starting work each day.
