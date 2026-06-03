from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PuliziaDataset:
    def __init__(self, dataset_id=9):
        self.__dataset = fetch_ucirepo(id=dataset_id)
        self.__X = self.__dataset.data.features
        self.__y = self.__dataset.data.targets
        self.__metadata = self.__dataset.metadata
        self.__variables = self.__dataset.variables

    def individua_na(self):
        return self.__X[self.__X.isna()].stack().index.tolist()

    def elimina_na(self):
        self.__X = self.__X.dropna()
        self.__y = self.__y.loc[self.__X.index]

    def trova_ed_elimina_na(self):
        posizioni_na = self.individua_na()
        self.elimina_na()
        return posizioni_na

    def trova_outliers(self):
        outliers = {}

        for col in self.__X.select_dtypes(include="number").columns:
            q1 = self.__X[col].quantile(0.25)
            q3 = self.__X[col].quantile(0.75)
            iqr = q3 - q1
            limite_basso = q1 - 1.5 * iqr
            limite_alto = q3 + 1.5 * iqr

            mask = (self.__X[col] < limite_basso) | (self.__X[col] > limite_alto)
            righe_outlier = self.__X[mask]

            if not righe_outlier.empty:
                outliers[col] = righe_outlier.index.tolist()

        return outliers

    def applica_pca(self):
        X_num = self.__X.select_dtypes(include="number")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)

        pca = PCA(n_components=3)
        componenti = pca.fit_transform(X_scaled)

        colonne_pca = ["PC1", "PC2", "PC3"]
        df_pca = pd.DataFrame(componenti, columns=colonne_pca, index=X_num.index)

        return df_pca, pca.explained_variance_ratio_

    def grafici_distribuzioni(self):
        X_num = self.__X.select_dtypes(include="number")
        X_num.hist(figsize=(12, 10), bins=20)
        plt.tight_layout()
        plt.show()
    def grafici_boxplot(self):
        X_num = self.__X.select_dtypes(include="number")
        plt.figure(figsize=(12, 8))
        X_num.boxplot(rot=45)
        plt.tight_layout()
        plt.show()
    def matrice_correlazione(self, metodo="pearson"):
        return self.__X.select_dtypes(include="number").corr(method=metodo)
    def jarque_bera_test(self):
        from scipy.stats import jarque_bera

        risultati = {}

        for col in self.__X.select_dtypes(include="number").columns:
            stat, p_value = jarque_bera(self.__X[col].dropna())
            risultati[col] = {
                "JB_stat": stat,
                "p_value": p_value,
                "normale": p_value > 0.05
            }

        return risultati

pulizia = PuliziaDataset()

print("NA trovati:", pulizia.individua_na())
pulizia.trova_ed_elimina_na()

print("Outliers:", pulizia.trova_outliers())
print("Matrice di correlazione:\n", pulizia.matrice_correlazione())

df_pca, varianza = pulizia.applica_pca()
print("PCA:")
print(df_pca)
print("Varianza spiegata:", varianza)

risultati_jb = pulizia.jarque_bera_test()
for colonna, info in risultati_jb.items():
    print(colonna, info)
