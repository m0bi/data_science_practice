import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    df = pd.read_csv('../../preprocessing/test_5_fold.csv')
    cat = [f for f in df.columns if 'cat' in f]
    num = [f for f in df.columns if 'cont' in f]
    df_cat = pd.DataFrame()
    for i in range(len(cat)):
        df_cat = pd.concat([df_cat, df[cat[i]]], axis=1)
    df_num = pd.DataFrame()
    for i in range(len(num)):
        df_num = pd.concat([df_num, df[num[i]]], axis=1)
    df_num = StandardScaler().fit_transform(df_num)
    df_cat = pd.get_dummies(df_cat)
    pca = PCA(n_components=10)
    df_num = pca.fit_transform(df_num)

    pca_cat = PCA(n_components=15)
    df_cat = pca_cat.fit_transform(df_cat)
    # print(pca_cat.explained_variance_ratio_[0:15])
    df_final = pd.concat([pd.DataFrame(df_cat), pd.DataFrame(
        df_num), df["loss"], df["kfold"]], axis=1)
    df_final.to_csv('test_5_fold_OHE.csv', index=False, sep=",")
