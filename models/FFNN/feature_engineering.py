import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocessing(path, name):
    df = pd.read_csv(path)
    # cat1 ... cat116
    y = df.loss
    kfolds = df.kfold
    cat = [f for f in df.columns if 'cat' in f]
    df_cat = pd.DataFrame()
    for i in range(len(cat)):
        df_cat = pd.concat([df_cat, df[cat[i]]], axis=1)
    df = df.drop("kfold", axis=1)
    df = df.drop("loss", axis=1)
    df = df.drop(cat, axis=1)
    pca_cat = PCA(n_components=18)
    df_cat = pd.get_dummies(df_cat)
    df_cat = pca_cat.fit_transform(df_cat)
    # print(df_cat.explained_variance_ratio_[10:20])
    ss = StandardScaler()
    df = ss.fit_transform(df)
    pca = PCA(n_components=10)
    df = pca.fit_transform(df)
    # print(df.explained_variance_ratio_[11:20])
    df = pd.concat([pd.DataFrame(df), pd.DataFrame(df_cat),
                    pd.Series(y), pd.Series(kfolds)], axis=1)
    df.to_csv(name, index=False, sep=",")


def preprocessingnok(path, name):
    df = pd.read_csv(path)
    # cat1 ... cat116
    y = df.loss
    cat = [f for f in df.columns if 'cat' in f]
    df_cat = pd.DataFrame()
    for i in range(len(cat)):
        df_cat = pd.concat([df_cat, df[cat[i]]], axis=1)
    df = df.drop("loss", axis=1)
    df = df.drop(cat, axis=1)
    pca_cat = PCA(n_components=18)
    df_cat = pd.get_dummies(df_cat)
    df_cat = pca_cat.fit_transform(df_cat)
    # print(df_cat.explained_variance_ratio_[10:20])
    ss = StandardScaler()
    df = ss.fit_transform(df)
    pca = PCA(n_components=10)
    df = pca.fit_transform(df)
    # print(df.explained_variance_ratio_[11:20])
    df = pd.concat([pd.DataFrame(df), pd.DataFrame(df_cat),
                    pd.Series(y)], axis=1)
    df.to_csv(name, index=False, sep=",")


if __name__ == "__main__":
    # preprocessing('../../preprocessing/test_5_fold.csv',
    #              'test_5_fold_SCALED.csv')
    preprocessingnok('../../preprocessing/hold_out.csv', 'hold_out_SCALED.csv')
