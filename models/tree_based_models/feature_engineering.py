import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA


def preprocessing(a, b):
    df = pd.read_csv(a)
    cat = [f for f in df.columns if 'cat' in f]
    num = [f for f in df.columns if 'cont' in f]
    # cat1 ... cat116
    y = df.loss
    kfolds = df.kfold
    df = df.drop("kfold", axis=1)
    df = df.drop("loss", axis=1)
    df_cat = pd.DataFrame()
    for i in range(len(cat)):
        df_cat = pd.concat([df_cat, df[cat[i]]], axis=1)
    df_num = pd.DataFrame()
    for i in range(len(num)):
        df_num = pd.concat([df_num, df[num[i]]], axis=1)
    df_cat = OrdinalEncoder().fit_transform(df_cat, y)
    pca_num = PCA(n_components=10)
    df_num = pca_num.fit_transform(df_num)
    # print(pca_num.explained_variance_ratio_[0:15])
    df_final = pd.concat([pd.DataFrame(df_cat), pd.DataFrame(df_num), pd.Series(y),
                          pd.Series(kfolds)], axis=1)
    df_final.to_csv(b, index=False, sep=",")


def preprocessing_final(a, b):
    df = pd.read_csv(a)
    cat = [f for f in df.columns if 'cat' in f]
    num = [f for f in df.columns if 'cont' in f]
    # cat1 ... cat116
    y = df.loss
    df = df.drop("loss", axis=1)
    df_cat = pd.DataFrame()
    for i in range(len(cat)):
        df_cat = pd.concat([df_cat, df[cat[i]]], axis=1)
    df_num = pd.DataFrame()
    for i in range(len(num)):
        df_num = pd.concat([df_num, df[num[i]]], axis=1)
    df_cat = OrdinalEncoder().fit_transform(df_cat, y)
    pca_num = PCA(n_components=10)
    df_num = pca_num.fit_transform(df_num)
    # print(pca_num.explained_variance_ratio_[0:15])
    df_final = pd.concat(
        [pd.DataFrame(df_cat), pd.DataFrame(df_num), pd.Series(y)], axis=1)
    df_final.to_csv(b, index=False, sep=",")


if __name__ == "__main__":
    # preprocessing('../../preprocessing/test_5_fold.csv', './test_5_fold_FE.csv'):
    preprocessing_final('../../preprocessing/hold_out.csv',
                        './hold_out_FINAL.csv')
