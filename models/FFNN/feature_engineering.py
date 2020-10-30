import pandas as pd
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    df = pd.read_csv('../../preprocessing/test_5_fold.csv')
    # cat1 ... cat116
    y = df.loss
    kfolds = df.kfold
    cat = []
    for i in range(116):
        cat.append(f'cat{i+1}')

    df_cat = pd.DataFrame()
    for i in range(116):
        df_cat = pd.concat([df_cat, df[cat[i]]], axis=1)
    df = df.drop("kfold", axis=1)
    df = df.drop("loss", axis=1)
    df = df.drop(cat, axis=1)
    df = df.drop("id", axis=1)
    df_cat = pd.get_dummies(df_cat)
    ss = StandardScaler()
    df = ss.fit_transform(df)
    df = pd.concat([pd.DataFrame(df), pd.DataFrame(df_cat),
                    pd.Series(y), pd.Series(kfolds)], axis=1)
    df.to_csv('test_5_fold_SCALED.csv', index=False, sep=",")
