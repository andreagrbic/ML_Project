from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import pairwise
import networkx as nx
from node2vec import Node2Vec


class Kernel(ABC):
    """
    Apstraktna klasa koja služi kao baza za implementaciju različitih kernela.
    Kerneli se koriste za generisanje kovarijacione matrice k_u.
    """

    @staticmethod
    def create_laplacian_matrix(edge_list: np.ndarray, n_nodes: int) -> np.ndarray:
        """
        Metoda računa Laplasovu matricu grafa korisnika i njihovih prijatelja.

        Parameters:
            edge_list: Lista čvorova grafa korisnika i njihovih prijatelja.
            n_nodes: Broj čvorova.

        Returns:
            Laplasova matrica L.
        """
        adjacency_matrix = np.zeros((n_nodes, n_nodes))
        for user, friend in edge_list:
            adjacency_matrix[user, friend] = 1
        return np.diag(np.sum(adjacency_matrix, axis=1)) - adjacency_matrix

    @abstractmethod
    def inv_kernel(self, edge_list: np.ndarray, n_nodes: int) -> np.ndarray:
        """
        Metoda računa odgovarajući kernel.

        Parameters:
            edge_list: Lista čvorova grafa korisnika i njihovih prijatelja.
            n_nodes: Broj čvorova.

        Returns:
            Matrica k_u dobijena odgovarajućim kernelom.
        """
        pass


class CommuteTimeKernel(Kernel):
    """
    Klasa koja opisuje grafovski kernel prosječnog vremena obilaska grafa (Commute Time kernel).
    """

    def __init__(self):
        pass

    def inv_kernel(self, edge_list: np.ndarray, n_nodes: int) -> np.ndarray:
        return np.linalg.pinv(self.create_laplacian_matrix(edge_list, n_nodes))


class RegularizedLaplacianKernel(Kernel):
    """
    Klasa koja opisuje regularizacioni Laplasov kernel.

    Parameters:
        gamma (float): (default = 0.95) regularizacioni hiperparametar
    """

    def __init__(self, gamma=0.95):
        self.gamma = gamma

    def inv_kernel(self, edge_list: np.ndarray, n_nodes: int) -> np.ndarray:
        L = self.create_laplacian_matrix(edge_list, n_nodes)
        return np.linalg.pinv(1 + self.gamma * L)


class DiffusionKernel(Kernel):
    """
    Klasa koja opisuje Difuzioni kernel (Diffusion Kernel).

    Parameters:
        beta (float): (default = 0.1) regularizacioni hiperparametar koji određuje stepen difuzije.
    """

    def __init__(self, beta=0.1):
        self.beta = beta

    def inv_kernel(self, edge_list: np.ndarray, n_nodes: int) -> np.ndarray:
        L = self.create_laplacian_matrix(edge_list, n_nodes)
        return np.exp(-self.beta * L)


class RBF(Kernel):
    """
    Klasa koja opisuje RBF kernel (Radial Basis Function Kernel).

    Parameters:
        dimensions (int): (default = 32) latentna dimenzija
        walk_length (int): (default = 20) Broj čvorova u svakom obilasku
        num_walks (int): (default = 100) Broj obilazaka po čvoru
        window (int): (default = 10) parametar modela
        min_count (int): (default = 1)  parametar modela
        batch_words (int): (default = 4) parametar modela
    """

    def __init__(self, window=10, min_count=1, batch_words=4, dimensions=32, walk_length=20, num_walks=100):

        self.params = {'window': window,
                       'min_count': min_count,
                       'batch_words': batch_words}
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks

    def inv_kernel(self, edge_list: np.ndarray, n_nodes: int) -> np.ndarray:
        # Kreiranje grafa
        G = nx.Graph()
        G.add_edges_from(edge_list)

        node2vec = Node2Vec(G, dimensions=self.dimensions, walk_length=self.walk_length, num_walks=self.num_walks)
        model = node2vec.fit(**self.params)

        # Vektorska reprezentacija podataka o korisnicima i njihovim prijateljima.
        embed_matrix = np.zeros((n_nodes, self.dimensions))
        for i in range(n_nodes):
            if str(i) in model.wv:
                embed_matrix[i] = model.wv[str(i)]
        return pairwise.rbf_kernel(embed_matrix)
