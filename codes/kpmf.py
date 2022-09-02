from typing import Tuple

import numpy as np

import pandas as pd
from codes.algorithm_base import AlgorithmBase
from codes.data import Data


class KPMF(AlgorithmBase):
    """
    Klasa u kojoj je implementirana metoda KPMF. Ukoliko su matrice k_u i k_v dijagonalne, KPMF se svodi na PMF.

    Parameters:
        learning_rate (float): (default = 0.005) hiperparametar koji određuje brzinu obučavanja modela
        regularization (float): (default = 0.02) regularizacioni hiperparametar koji određuje stepen regularizacije
        n_epochs (int): (default = 20) broj epoha
        n_factors (int): broj latentnih faktora
        kernel (Kernel): (default = 'ct') parametar kojim se određuje izbor kernela
        kernel_kwargs (dict): (default = None) rječnik hiperparametara klase Kernel; npr, ukoliko je odabrano jezgro
         diff (Diffusion Kernel), može da uzima vrednost {'beta': 0.2}
    """

    def __init__(self, learning_rate=0.005, regularization=0.02, n_epochs=20, n_factors=10, kernel='ct',
                 kernel_kwargs=None):
        super().__init__(learning_rate, regularization, n_epochs, n_factors, kernel, kernel_kwargs)

    def initialize(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:

        u = np.random.multivariate_normal(mean=np.zeros(shape=data.n_users), cov=self.k_u(data), size=self.n_factors).T
        v = np.random.multivariate_normal(mean=np.zeros(shape=data.n_items), cov=self.k_v(data), size=self.n_factors).T

        return u, v

    def train_model(self, data_train: Data, df_user_item_val: pd.DataFrame, verbose: bool):

        list_train_rmse = []
        list_val_rmse = []

        print('Inicijalizacija u i v...')
        u, v = self.initialize(data_train)

        print('Ucenje...')
        for epoch_ix in range(self.n_epochs):
            if verbose:
                print(f'Epoha {epoch_ix + 1}:')
            u, v = self.run_epoch(data_train, u, v)
            train_rmse = self.compute_metrics(data_train.user_item_data, u, v)
            val_rmse = self.compute_metrics(df_user_item_val, u, v)

            if verbose:
                print(f'train loss: {train_rmse}; val loss:{val_rmse}')

            list_train_rmse.append(train_rmse)
            list_val_rmse.append(val_rmse)

        return u, v, list_train_rmse, list_val_rmse

    def run_epoch(self, data: Data, u: np.ndarray, v: np.ndarray, y: np.ndarray = None, w: np.ndarray = None) \
            -> Tuple[np.ndarray, np.ndarray]:

        _, n_user_rated, n_item_rated = data.create_indicator_matrix()
        s_u = self.s_u(data)
        s_v = self.s_v(data)

        for i, row in data.user_item_data.iterrows():
            user = int(row['user'])
            item = int(row['item'])
            rating = row['rating']

            # Predikcija rejtinga
            pred = np.dot(u[user], v[item])
            err = rating - pred

            # Ažuriranje latentnih faktora
            u_current = u[user]
            u[user] += self.lr * (err * v[item] - self.reg / n_user_rated[user] / 2 * (s_u[user] @ u + u[user]))
            v[item] += self.lr * (err * u_current - self.reg / n_item_rated[item] / 2 * (s_v[item] @ v + v[item]))

        return u, v
