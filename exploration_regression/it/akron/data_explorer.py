from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import jarque_bera
import numpy as np
from io import BytesIO




class PuliziaDataset:
    def __init__(self, dataset_id=9):
        self.__dataset = fetch_ucirepo(id=dataset_id)
        self.__X = self.__dataset.data.features
        self.__y = self.__dataset.data.targets

    def replace_strange_missing_values(self):
        # cerca valori strani di missing values e sostituiscili con nan
        strange_tokens = ["?", "Unknown", "unknown", "NAN", "nan", "NA", "na", "", " "]
        return self.__X.replace(strange_tokens, np.nan)

    def individua_na(self):
        df = self.replace_strange_missing_values()

        null_values = df.isna().sum()
        miss_percent = df.isna().sum() * 100 / len(df)
        return null_values, miss_percent

    def elimina_na(self):
        df = self.replace_strange_missing_values()
        X = df.dropna()
        y = self.__y.loc[X.index]
        return X, y


    def trova_outliers(self):
        df, y = self.elimina_na()

        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        y_final = y.loc[df.index]

        return df, y_final

    def applica_pca(self):
        df = self.replace_strange_missing_values()
        X_num = df.select_dtypes(include="number").dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)

        pca = PCA(n_components=X_scaled.shape[1])
        pca.fit(X_scaled)

        fig, ax = plt.subplots()

        ax.plot(np.cumsum(pca.explained_variance_ratio_))
        ax.set_xlabel('Components')
        ax.set_ylabel('Explained variance (cumulative)')
        ax.set_title('PCA Explained Variance')

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        plt.close(fig)

        return buf

    def grafici_distribuzioni(self):
        X_num = self.__X.select_dtypes(include="number")

        fig = X_num.hist(figsize=(12, 10), bins=20)

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        plt.close()

        return buf

    def grafici_boxplot(self):
        X_num = self.__X.select_dtypes(include="number")

        fig, ax = plt.subplots(figsize=(12, 8))
        X_num.boxplot(rot=45, ax=ax)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        plt.close(fig)

        return buf

    def matrice_correlazione(self, metodo="pearson"):
        df = self.replace_strange_missing_values()
        return df.select_dtypes(include="number").corr(method=metodo)
    
    def grafici_correlazione(self, metodo="pearson"):
        df = self.replace_strange_missing_values()

        corr = df.select_dtypes(include="number").corr(method=metodo)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            corr,
            annot=True,  # mostra i valori
            fmt=".2f",  # 2 decimali
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )

        ax.set_title(f"Matrice di Correlazione ({metodo.title()})")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        plt.close(fig)

        return buf

    def jarque_bera_test(self):
        df = self.replace_strange_missing_values()

        results = {}
        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            data = pd.to_numeric(df[col], errors='coerce').dropna()

            if len(data) < 8:
                continue

            if data.nunique() < 2:
                continue

            stat, p = jarque_bera(data)

            results[col] = {
                "stat": float(stat),
                "p_value": float(p),
                "normale": bool(p > 0.05)
            }

        return results
