from typing import Tuple
import numpy as np
import pandas as pd
from codes.algorithm_base import AlgorithmBase
from codes.data import Data


class CKPMF(AlgorithmBase):
    """
    Klasa u kojoj je implementirana metoda CKPMF. Ukoliko je matrica k_v dijagonalna, CKPMF se svodi na CPMF.

    Parameters:
        learning_rate (float): (default = 0.005) hiperparametar koji brzinu obučavanja modela.
        regularization (float): (default = 0.02) regularizacioni hiperparametar koji određuje stepen regularizacije
        n_epochs (int): (default = 20) broj epoha
        n_factors (int): broj latentnih faktora

    """
    def __init__(self, learning_rate=0.005, regularization=0.02, n_epochs=20, n_factors=100):
        super().__init__(learning_rate, regularization, n_epochs, n_factors, kernel='ckpmf')

    def initialize(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros(shape=(data.n_users, self.n_factors))
        v = np.random.multivariate_normal(mean=np.zeros(shape=data.n_items), cov=self.k_v(data), size=self.n_factors).T
        return u, v

    def initialize_y_w(self, n_items: int, n_users: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Metoda koja vrši inicijalizaciju pomoćnih matrica W i Y u metodi CKPMF. One se generišu iz normalne raspodjele.

        Parameters:
            n_items: Broj izvođača
            n_users: Broj korisnika
        Returns:
            Matrice Y i W.
        """
        return np.random.normal(0, 0.1, (n_users, self.n_factors)), np.random.normal(0, 0.1, (n_items, self.n_factors))

    def train_model(self, data_train: Data, df_user_item_val: pd.DataFrame, verbose: bool):
        list_train_rmse = []
        list_val_rmse = []

        print('Inicijalizacija u, v, y i w...')
        u, v = self.initialize(data_train)
        y, w = self.initialize_y_w(data_train.n_items, data_train.n_users)

        print('Ucenje...')
        for epoch_ix in range(self.n_epochs):
            if verbose:
                print(f'Epoha {epoch_ix + 1}:')
            u, v, y, w = self.run_epoch(data_train, u, v, y, w)
            train_rmse = self.compute_metrics(data_train.user_item_data, u, v)
            val_rmse = self.compute_metrics(df_user_item_val, u, v)

            if verbose:
                print(f'train loss: {train_rmse}; val loss:{val_rmse}')

            list_train_rmse.append(train_rmse)
            list_val_rmse.append(val_rmse)

        return u, v, list_train_rmse, list_val_rmse

    def run_epoch(self, data: Data, u: np.ndarray, v: np.ndarray, y: np.ndarray = None, w: np.ndarray = None):
        indicator_matrix, n_user_rated, n_item_rated = data.create_indicator_matrix()
        s_v = self.s_v(data)

        for i, row in data.user_item_data.iterrows():
            user = int(row['user'])
            item = int(row['item'])
            rating = row['rating']

            u[user] = y[user] + (indicator_matrix[user] @ w) / n_user_rated[user]

            # Predikcija rejtinga
            pred = np.dot(u[user], v[item])
            err = rating - pred

            # Ažuriranje latentnih faktora
            v_i = v[item]
            w_i = w[item]
            y[user] += self.lr * (err * v[item] - self.reg / n_user_rated[user] * y[user])
            v[item] += self.lr * (err * u[user] - self.reg / n_item_rated[item] / 2 * (s_v[item] @ v + v[item]))
            w += self.lr * (err * np.outer(indicator_matrix[user], v_i))
            w[item] -= self.lr * self.reg / n_item_rated[item] * w_i
        return u, v, y, w
