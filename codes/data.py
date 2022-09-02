from typing import Tuple

import numpy as np
import pandas as pd


class Data:
    """
    Klasa u kojoj se vrši obrada podataka.

    Parameters:
        df_user_item:  Data Frame koji sadrži informacije o korisnicima i izvođačima.
        df_user_friends: Data Frame koji sadrži informacije o korisnicima i njihovim prijateljima.
        df_user_tagged_item: Data Frame koji sadrži informacije o izvođačima i dodijeljenim tagovima (žanrovima muzike).
    """

    def __init__(self, df_user_item: pd.DataFrame, df_user_friends: pd.DataFrame, df_user_tagged_item: pd.DataFrame):

        self.user_dict = self.get_user_dict(df_user_item)
        self.item_dict = self.get_item_dict(df_user_item)
        self.n_users = len(self.user_dict)
        self.n_items = len(self.item_dict)

        self.user_item_data = self.preprocess_user_item_data(df_user_item)
        self.user_friends_data = self.preprocess_user_friends_data(df_user_friends)
        self.user_tagged_item_data = self.preprocess_user_tagged_item_data(df_user_tagged_item)

    @staticmethod
    def get_user_dict(X: pd.DataFrame) -> dict:
        """
        Metoda koja generiše rječnik korisnika. Izdvajaju se svi korisnici koji se pojavljuju u skupu
        i svakom od njih se dodjeljuje jedinstven redni broj čime se kreira rječnik korisnika.

        Parameters:
            X: Data Frame koji sadrži informacije o korisnicima i izvođačima
        Returns:
            user_dict: Rječnik korisnika.
        """
        user_id = X['userID'].unique().tolist()
        n_users = len(user_id)
        user_dict = dict(zip(user_id, [i for i in range(n_users)]))
        return user_dict

    @staticmethod
    def get_item_dict(X: pd.DataFrame) -> dict:
        """
        Metoda koja generiše rječnik izvođača. Izdvajaju se svi izvođači koji se pojavljuju u skupu i svakom od njih se
        dodjeljuje jedinstven redni broj čime se kreira rječnik izvođača.

        Parameters:
            X: Data Frame koji sadrži informacije o korisnicima i izvođačima
        Returns:
            user_dict: Rječnik izvođača.
        """
        artist_id = X['artistID'].unique().tolist()
        n_items = len(artist_id)
        item_dict = dict(zip(artist_id, [i for i in range(n_items)]))
        return item_dict

    def preprocess_user_item_data(self, df_user_item: pd.DataFrame):
        """
        Metoda koja vrši preprocesiranje podataka o korisnicima i izvođačima.
        Korisnicima i izvođačima se mapiranjem dodijele rječnici korisnika i izvođača.

        Ukoliko ne postoji informacija o rejtingu za par korisnik/izvođač, onda se rejting postavlja na -1.

        Rejting se dobija primjenom logaritamske transformacije na kolonu 'weight'.

        Parameters:
            df_user_item: Podaci o korisnicima i izvođačima
        Returns:
            Data Frame koji sadrži podatke o korisnicima i izvođačima mapirane odgovarajućim rječnicima i rejtingu.
        """
        X = df_user_item.copy()
        X.loc[:, "user"] = X["userID"].map(self.user_dict).fillna(-1).astype(np.int32)
        X.loc[:, "item"] = X["artistID"].map(self.item_dict).fillna(-1).astype(np.int32)

        X.loc[:, 'rating'] = np.log(X['weight'])
        return X[["user", "item", "rating"]]

    def preprocess_user_friends_data(self, df_user_friends: pd.DataFrame) -> np.ndarray:
        """
        Metoda koja vrši preprocesiranje podataka o korisnicima i njihovim prijateljima.
        Izdvajamo one korisnike i njihove prijatelje koji se nalaze u kreiranom rječniku korisnika.
        Potom podacima mapiranjem dodjeljujemo jedinstvene redne brojeve iz rječnika korisnika i izvođačima.

        Parameters:
            df_user_friends: Podaci o društvenoj mreži korisnika
        Returns:
            Matrica koja sadrži podatke o korisnicima i njihovim prijateljima mapiranim odgovarajućim
            rječnicima
        """
        X = df_user_friends[
            df_user_friends['userID'].isin(self.user_dict) & df_user_friends['friendID'].isin(self.user_dict)]
        X.loc[:, 'userID'] = X['userID'].map(self.user_dict)
        X.loc[:, 'friendID'] = X['friendID'].map(self.user_dict)
        user_edge_list = X.values.astype(np.int32)
        return user_edge_list

    def preprocess_user_tagged_item_data(self, df_user_tagged_item: pd.DataFrame) -> np.ndarray:
        """
        Metoda koja vrši preprocesiranje podataka o dodijeljenim tagovima izvođačima od strane korisnika.
        Izdvajamo izvođače koji se nalaze u kreiranom rječniku izvođača. Izdvajamo tagove (žanrove muzike) i kreiramo
        rječnik tagova. Podacima o tagovima i izvođačima dodjeljujemo jedinstvene redne brojeve mapiranjem pomoću
        odgovarajućih rječnika.
        Kreiramo matricu koja sadrži informaciju o tome da li je određeni tag (žanr muzike) dodijeljen izvođaču,
        ili nije. Ako tag jeste dodijeljen, onda se u tom polju matrice nalazi 1, a inače 0.

        Parameters:
            df_user_tagged_item: Podaci o dodijeljenim tagovima.
        Returns:
            Matrica koja sadrži informaciju o tome da li je određeni tag (žanr muzike) dodijeljen izvođaču, ili nije.
        """
        X = df_user_tagged_item[df_user_tagged_item['artistID'].isin(self.item_dict)][['artistID', 'tagID']]
        tag_id = X['tagID'].unique().tolist()
        n_tags = len(tag_id)

        tag_dict = dict(zip(tag_id, [i for i in range(n_tags)]))
        X.loc[:, 'tagID'] = X['tagID'].map(tag_dict)
        X.loc[:, 'artistID'] = X['artistID'].map(self.item_dict)

        artist_tag_matrix = np.zeros((self.n_items, n_tags))
        for artist, tag in X.values.astype(np.int32):
            artist_tag_matrix[artist, tag] = 1
        return artist_tag_matrix

    def create_indicator_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Metoda koja računa indikatorsku matricu I koja sadrži informaciju o tome da li je određeni tag(žanr muzike)
        dodijeljen izvođaču, ili nije.
        Matrica I se kreira tako da sadrži jedinicu na svakom polju gdje je korisnik dao rejting izvođaču.

        Returns:
            Indikatorska matrica I, broj ocjena koje je svaki korisnik dodijelio, broj ocjena koje je svaki izvođač dobio.
        """
        X = self.user_item_data.values
        I = np.zeros((self.n_users, self.n_items))
        for user, item in X[:, :2]:
            I[int(user), int(item)] = 1

        n_user_rated = np.sum(I, axis=1)
        n_item_rated = np.sum(I, axis=0)
        return I, n_user_rated, n_item_rated
