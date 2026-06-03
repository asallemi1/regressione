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


class Regressione:
    def __init__(self, X, y, metodo="lineare", alpha=1.0, n_components=3):
        self.__X = X
        self.__y = y
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__X_train_pca = None
        self.__X_test_pca = None
        self.__y_pred = None
        self.__mse = None
        self.__metodo = metodo.lower()
        self.__alpha = alpha
        self.__n_components = n_components

        self.__scaler = StandardScaler()
        self.__pca = PCA(n_components=self.__n_components)
        self.__modello = self.__crea_modello()

    def __crea_modello(self):
        if self.__metodo == "lineare":
            return LinearRegression()

        elif self.__metodo == "ridge":
            return Ridge(alpha=self.__alpha)

        elif self.__metodo == "lasso":
            return Lasso(alpha=self.__alpha)

        elif self.__metodo == "elasticnet":
            return ElasticNet(alpha=self.__alpha, l1_ratio=0.5)

        else:
            raise ValueError(
                "Metodo non valido. Usa: 'lineare', 'ridge', 'lasso' oppure 'elasticnet'"
            )

    def splitta_dataset(self, test_size=0.2, random_state=42):
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            self.__X,
            self.__y,
            test_size=test_size,
            random_state=random_state
        )
        return self.__X_train, self.__X_test, self.__y_train, self.__y_test

    def applica_scaling_e_pca(self):
        X_train_scaled = self.__scaler.fit_transform(self.__X_train)
        X_test_scaled = self.__scaler.transform(self.__X_test)

        self.__X_train_pca = self.__pca.fit_transform(X_train_scaled)
        self.__X_test_pca = self.__pca.transform(X_test_scaled)

        return self.__X_train_pca, self.__X_test_pca, self.__pca.explained_variance_ratio_

    def addestra_e_valuta(self):
        self.__modello.fit(self.__X_train_pca, self.__y_train)
        self.__y_pred = self.__modello.predict(self.__X_test_pca)
        self.__mse = mean_squared_error(self.__y_test, self.__y_pred)
        return self.__mse

    def mostra_risultati(self):
        print("=" * 60)
        print("REGRESSIONE CON PCA")
        print("=" * 60)
        print(f"Metodo scelto: {self.__metodo}")
        print(f"Alpha: {self.__alpha}")
        print(f"Numero componenti PCA: {self.__n_components}")
        print(f"Varianza spiegata: {sum(self.__pca.explained_variance_ratio_)*100:.2f}%")
        print(f"\nMSE sul test set: {self.__mse:.4f}")
        print(f"RMSE: {self.__mse**0.5:.4f}")

        print(f"\nPredizioni sul test set:")
        print(self.__y_pred)

        print(f"\nCoefficients:")
        coef = self.__modello.coef_

        if np.ndim(coef) == 1:
            for i, c in enumerate(coef):
                print(f"  PC{i+1}: {c:.4f}")
        else:
            for target_idx, row in enumerate(coef):
                print(f"  Target {target_idx + 1}:")
                for i, c in enumerate(row):
                    print(f"    PC{i+1}: {c:.4f}")

        print("\nIntercept:")
        intercept = self.__modello.intercept_
        if np.ndim(intercept) == 0:
            print(f"  {intercept:.4f}")
        else:
            for i, val in enumerate(np.ravel(intercept)):
                print(f"  Target {i+1}: {val:.4f}")

    def get_predizioni(self):
        return self.__y_pred

    def get_mse(self):
        return self.__mse

    def get_modello(self):
        return self.__modello

