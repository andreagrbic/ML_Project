from abc import abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from codes.kernels import CommuteTimeKernel, RegularizedLaplacianKernel, DiffusionKernel, RBF
from codes.data import Data


class AlgorithmBase(object):
    """
    Apstraktna klasa za modele Kernelizovanih Probabilističkih Matričnih Faktorizacija.
    U njoj su implementirane funkcije fit, predict i izvršena je inicijalizacija matrica k_u, k_v.
    Ima dvije nasijeđene klase, KPMF i CKPMF.

    Parameters:
        learning_rate (float): (default = 0.005) hiperparametar koji određuje brzinu obučavanja modela
        regularization (float): (default = 0.02) regularizacioni hiperparametar koji određuje stepen regularizacije
        n_epochs (int): (default = 20) broj epoha
        n_factors (int): (default = 10) broj latentnih faktora
        kernel (Kernel): (default = 'ct') parametar kojim se određuje izbor kernela; mora uzimati jednu od vrednosti
         'ct', 'reg', 'rbf', 'dif', 'ckpmf'. Ukoliko je vrijednost kernela 'ckpmf' to označava da radimo sa metodom
         CKPMF u kojoj se ne koriste kerneli, odnosno koristi se samo kovarijaciona funkcija koju ne navodimo
         kao poseban kernel.
        kernel_kwargs (dict): (default = None) rječnik hiperparametara klase Kernel; npr, ukoliko je odabrano jezgro
         diff (Diffusion Kernel), može da uzima vrednost {'beta': 0.2}
    """

    def __init__(self, learning_rate=0.005, regularization=0.02, n_epochs=20, n_factors=10, kernel='ct',
                 kernel_kwargs=None):
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs
        self.n_factors = n_factors

        self.avg_rating = 0

        self.u = None
        self.v = None

        self.user_dict = None
        self.item_dict = None

        self.train_loss = None
        self.val_loss = None

        self.train_loss_per_epoch = None
        self.val_loss_per_epoch = None

        self.is_trained = False

        assert kernel in ['ct', 'reg', 'rbf', 'dif', 'ckpmf'], 'Parametar kernel mora uzimati jednu od vrednosti ' \
                                                               '"ct" (Commute Time), "reg" (Regularized Laplacian), ' \
                                                               '"rbf" (Radial Basis Function), "dif" (Diffusion), ili' \
                                                               '"ckpmf" (CKPMF metoda nema kernele)!'
        # Izbor kernela.
        if kernel == 'ct':
            self.kernel = CommuteTimeKernel()
        elif kernel == 'reg':
            if kernel_kwargs is not None:
                self.kernel = RegularizedLaplacianKernel(**kernel_kwargs)
            else:
                self.kernel = RegularizedLaplacianKernel()
        elif kernel == 'rbf':
            if kernel_kwargs is not None:
                self.kernel = RBF(**kernel_kwargs)
            else:
                self.kernel = RBF()
        elif kernel == 'dif':
            if kernel_kwargs is not None:
                self.kernel = DiffusionKernel(**kernel_kwargs)
            else:
                self.kernel = DiffusionKernel()
        else:
            self.kernel = None

    @abstractmethod
    def initialize(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Metoda vrši inicijalizaciju matrica U i V (njihovi elementi su parametri modela).

        Parameters:
            data: Skup podataka klase Data.
        Returns:
            Inicijalizovane matrice U i V.
        """
        pass

    @abstractmethod
    def run_epoch(self, data: Data, u: np.ndarray, v: np.ndarray, y: np.ndarray = None, w: np.ndarray = None):
        """
        Metoda vrši jednu epohu obučavanja modela. U njoj se računa predikcija rejtinga, greška i ažuriraju se
        parametri modela (u,v).

        Parameters:
                data: Skup podataka klase Data.
                u: Matrica latentnih faktora korisnika U.
                v: Matrica latentnih faktora izvođača.
                y: (default=None) Pomoćna matrica za generisanje matrice U. Koristi se samo u metodi CKPMF.
                w: (default=None) Pomoćna matrica za generisanje matrice U. Koristi se samo u metodi CKPMF.
        Returns:
             Matrice U i V nakon izvršene epohe ukoliko se radi o KPMF metodi, a ukoliko se radi o CKPMF metodi vraća
              U, V, W, Y.
        """
        pass

    @abstractmethod
    def train_model(self, data_train: Data, df_user_item_val: pd.DataFrame, verbose: bool):

        """
        Metoda koja se koristi prilikom obučavanja modela. U njoj se inicijalizuju matrice U, V, obučava model, računaju
        greške na skupu za trening i validaciju i računaju vrijednosti matrica U i V.

        Parameters:
           data_train: Trening skup (sadrži informacije o korisnicima i izvođačima)
           df_user_item_val: Validacioni skup (sadrži informacije o korisnicima i izvođačima)
           verbose: Indikator da li je potrebno štampati rezultate svih epoha tokom obučavanja modela

        Returns:
           Finalne matrice U i V nakon obučavanja, liste sa greškama na skupu za trening i validaciju po epohama
        """
        pass

    def k_u(self, data: Data) -> np.ndarray:
        """
        Metoda vrši generisanje matrice kovarijacije k_u korišćenjem odgovarajućeg jezgra.

        Parameters:
            data: Skup podataka klase Data.
        Returns:
            Generisana matrica k_u.
        """
        return self.kernel.inv_kernel(data.user_friends_data, data.n_users)

    @staticmethod
    def k_v(data: Data) -> np.ndarray:
        """
        Metoda vrši generisanje matrice kovarijacije k_v.

        Parameters:
            data: Skup podataka klase Data.
        Returns:
            Generisana matrica k_v.
        """
        return np.cov(data.user_tagged_item_data)

    def s_u(self, data: Data) -> np.ndarray:
        """
        Metoda vrši računanje matrice s_u (inverz matrice k_u).

        Parameters:
            data: Skup podataka klase Data.
        Returns:
            Matrica s_u.
        """
        return np.linalg.pinv(self.k_u(data))

    def s_v(self, data: Data) -> np.ndarray:
        """
        Metoda vrši računanje matrice s_v (inverz matrice k_v).

        Parameters:
            data: Skup podataka klase Data.

        Returns:
            Matrica s_v.
        """
        return np.linalg.pinv(self.k_v(data))

    def fit(self, df_user_item_train: pd.DataFrame, df_user_item_val: pd.DataFrame, df_user_friend: pd.DataFrame,
            df_user_tagged_item: pd.DataFrame, verbose=True) -> None:
        """
        Metoda vrši obučavanje modela i izračunavanje grešaka na skupu za trening i validaciju.

        Parameters:
            df_user_item_train: Trening skup korisnika i izvođača.
            df_user_item_val: Validacioni skup korisnika i izvođača.
            df_user_friend: Podaci o društvenoj mreži korisnika.
            df_user_tagged_item: Podaci o tagovima koji su dodijeljeni izvođačima.
            verbose: (default = True) Indikator koji označava da li se štampaju detalji učenja.
        """

        print('Obrada podataka...')
        data_train = Data(df_user_item_train, df_user_friend, df_user_tagged_item)
        df_user_item_val = data_train.preprocess_user_item_data(df_user_item_val)
        self.user_dict = data_train.user_dict
        self.item_dict = data_train.item_dict
        self.avg_rating = np.mean(data_train.user_item_data['rating'].values)

        u, v, list_train_rmse, list_val_rmse = self.train_model(data_train, df_user_item_val, verbose)
        self.u, self.v = u, v
        self.train_loss_per_epoch = list_train_rmse
        self.val_loss_per_epoch = list_val_rmse
        self.train_loss = list_train_rmse[-1]
        self.val_loss = list_val_rmse[-1]
        self.is_trained = True

    def compute_metrics(self, df_user_item: pd.DataFrame, u: np.ndarray, v: np.ndarray, ) -> float:
        """
        Metoda računa grešku RMSE na datom skupu podataka. Vrijednosti se predviđaju na osnovu datih matrica U i V.

        Parameters:
            df_user_item: Skup podataka korisnika i izvođača.
            u: Matrica U.
            v: Matrica V.
        Returns:
            rmse: greška RMSE
        """
        residuals = []

        for _, row in df_user_item.iterrows():
            user = int(row['user'])
            item = int(row['item'])
            rating = row['rating']

            # predict global mean if user or item is new
            pred = self.avg_rating

            if (user > -1) and (item > -1):
                pred = np.dot(u[user], v[item])

            residuals.append(rating - pred)

        residuals = np.array(residuals)
        loss = np.square(residuals).mean()
        rmse = np.sqrt(loss)

        return rmse

    def predict_pair(self, user_id: int, item_id: int) -> float:
        """
        Metoda vraća predikciju modela za datog korisnika i izvođača.
        U slučaju da u rječniku ne postoji dati korisnik ili izvođač, predikcija je prosječna vrijednost rejtinga.

        Parameters:
            user_id: ID korisnika (u originalnom Df-u)
            item_id: ID izvođača (u originalnom Df-u)
        Returns:
            pred: Predikcija rejtinga za datog korisnika i izvođača.
        """

        if (user_id in self.user_dict) and (item_id in self.item_dict):
            user_index = self.user_dict[user_id]
            item_index = self.item_dict[item_id]
            pred = np.dot(self.u[user_index], self.v[item_index])
        else:
            pred = self.avg_rating

        return pred

    def predict(self, df_user_item_test: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Metoda vrši predikciju modela i izračunava grešku na test skupu.

        Parameters:
            df_user_item_test: Skup izvođača i korisnika za koje se vrši predikcija rejtinga.
        Returns:
             Matrica predviđenih rejtinga i greška na skupu za testiranje.
        """
        assert self.is_trained, 'Prije pozivanja metode predict potrebno je metodom fit istrenirati model.'

        df_user_item_pred = df_user_item_test.copy()
        df_user_item_pred.loc[:, 'rating'] = np.log(df_user_item_pred['weight'])

        df_user_item_pred.loc[:, 'prediction'] = df_user_item_pred.apply(
            lambda x: self.predict_pair(x['userID'], x['artistID']), axis=1)

        test_loss = np.sqrt(np.mean(np.square(df_user_item_pred['rating'] - df_user_item_pred['prediction'])))
        return df_user_item_pred, test_loss
