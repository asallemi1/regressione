from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Regressione:
    def __init__(self, X, y, metodo="elasticnet"):
        self.__X = X
        self.__y = y
        self.__model = None
        self.__scaler = None
        self.__pca = None
        self.__metodo = metodo

    def train(self):
        X_train_pca, X_test_pca, y_train, y_test, scaler, pca = self.applica_scaling_e_pca()

        self.__scaler = scaler
        self.__pca = pca

        if self.__metodo == "lineare":
            self.__model = LinearRegression()

        elif self.__metodo == "ridge":
            self.__model = Ridge()

        elif self.__metodo == "lasso":
            self.__model = Lasso()

        else:
            self.__model = ElasticNet(l1_ratio=0.5)

        self.__model.fit(X_train_pca, y_train)

        y_pred = self.__model.predict(X_test_pca)
        mse = mean_squared_error(y_test, y_pred)

        return mse


    def splitta_dataset(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.__X,
            self.__y,
            test_size=test_size,
            random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def applica_scaling_e_pca(self):
        X_train, X_test, y_train, y_test = self.splitta_dataset()

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=3)

        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        X_train_pca = pd.DataFrame(
            X_train_pca,
            columns=["PC1", "PC2", "PC3"],
            index=X_train.index
        )

        X_test_pca = pd.DataFrame(
            X_test_pca,
            columns=["PC1", "PC2", "PC3"],
            index=X_test.index
        )

        return X_train_pca, X_test_pca, y_train, y_test,  scaler, pca

    def new_pred(self, new_data):
        new_data = new_data[self.__X.columns]

        new_scaled = self.__scaler.transform(new_data)
        new_pca = self.__pca.transform(new_scaled)

        return self.__model.predict(new_pca)

