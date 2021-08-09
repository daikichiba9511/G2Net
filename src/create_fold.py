import pandas as pd
from sklearn import model_selection

SEED: int = 42


def create_fold(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df_ = df.copy()
    target = df_["target"].values
    data = df_.drop("target", axis=1)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    for idx, (train_idx, valid_idx) in enumerate(kf.split(data, target)):
        df_.loc[valid_idx, "fold"] = idx
    df_["fold"] = df_["fold"].astype("int")
    return df_


df = pd.read_csv("./input/g2net-gravitational-wave-detection/training_labels.csv")
folded_df = create_fold(df)
print(folded_df.head(10))
print(folded_df[["fold"]].value_counts())
folded_df.to_csv("./input/g2net-gravitational-wave-detection/fold_train_df.csv", index=False)