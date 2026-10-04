import pandas as pd

TARGET = "fault_types"
FEATURES = ["Ia","Ib","Ic","Va","Vb","Vc","Im","Vm","Pai","Pav","Fi","Fv"]
CLASSES = ["NF","LG","LL","LLG","LLL","LLLG"]
RANDOM_STATE = 42

def load_clean(path="data/processed/clean.csv"):
    return pd.read_csv(path)

def blocked_split(df, n_blocks=5):
    df = df.copy()
    n = df.groupby(TARGET)[TARGET].transform("size")
    df["block"] = df.groupby(TARGET).cumcount() * n_blocks // n
    train = df[df["block"] < n_blocks - 1]   # first 80% of every run
    test = df[df["block"] == n_blocks - 1]   # last 20% of every run
    return train, test
