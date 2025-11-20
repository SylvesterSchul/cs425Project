# Setup Instructions

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Download Dataset
```bash
cd data/raw
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ../..
```

**Windows:** Download from https://grouplens.org/datasets/movielens/100k/ and extract to `data/raw/`

## 3. Run the System
```bash
python src/main.py
```

## Project Structure
```
cs425Project/
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── collaborative_filtering.py
│   ├── minhash_lsh.py
│   ├── evaluation.py
│   └── main.py
├── data/
│   ├── raw/               # MovieLens dataset goes here
│   └── processed/         # Processed data
├── docs/
│   └── proposal.md        # Project proposal
├── results/               # Output graphs/results
├── requirements.txt
└── .gitignore
```

## Work Division

**Student 1:** Implement `data_preprocessing.py` and `collaborative_filtering.py`
**Student 2:** Implement `minhash_lsh.py`  
**Student 3:** Implement `evaluation.py` and create presentation
