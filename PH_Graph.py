import numpy as np
import gudhi as gd
import sklearn.metrics as ms
from sklearn.neighbors import radius_neighbors_graph
import warnings


warnings.filterwarnings("ignore")


def find_index_greater_than_or_equal_to(values, target):
    for i, value in enumerate(values):
        if value >= target:
            return i + 1
    return 0


class PHGraph:
    def __init__(self, rate=0.9, max_value=1.1, max_dimension=1, verbose=False):
        self.rate = rate
        self.max_value = max_value
        self.max_dimension = max_dimension
        self.verbose = verbose

    def computePH(self, X, Alpha=True):
        rips = gd.RipsComplex(X)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension)
        dgms = simplex_tree.persistence()
        H0 = np.array([*np.array(dgms[::-1])[:, 1]])
        return H0

    def interval(self, H0):
        death_values = H0[:, 1]
        unique_death_values = np.unique(death_values)
        sorted_death_values = np.sort(unique_death_values)

        finite_death_values = sorted_death_values[sorted_death_values != np.inf]
        max_finite = finite_death_values[-1]
        inf_value_w = max_finite * self.max_value
        adjusted_death_values = np.append(finite_death_values, inf_value_w)

        survival_difference = np.diff(np.insert(adjusted_death_values, 0, 0))
        para = (
            np.insert(adjusted_death_values, 0, 0)[: len(survival_difference)]
            + survival_difference / 2
        )
        epi_up = (
            np.insert(adjusted_death_values, 0, 0)[: len(survival_difference)]
            + survival_difference
        )
        return survival_difference, para[1:], epi_up

    def select_epi_rate(self, epi, persistence, gamma):
        w = np.power(persistence, 2)
        rate_set = [np.sum(w[:i]) / np.sum(w) for i in range(len(epi))]

        min_index = find_index_greater_than_or_equal_to(rate_set, self.rate)
        epi = epi[:min_index]
        weight = w[:min_index] / np.sum(w[:min_index])
        gamma = gamma[:min_index]

        if self.verbose:
            print("new Cumulative contribution rate: %.2f" % (np.sum(w[:min_index]) / np.sum(w)))
            print(epi)
            print(len(epi))
            print(gamma)
        return epi, weight, gamma

    def majorPara(self, survival_difference, para, epi_up):
        TI = [
            survival_difference[i] / (np.sum(survival_difference[:i]) / i)
            for i in range(1, len(survival_difference))
        ]
        index = np.argsort(-np.array(TI))
        epi_sort = [para[index[i]] for i in range(len(index))]
        TI_sort = [TI[index[i]] for i in range(len(index))]
        gamma_sort = [epi_up[index[i]] for i in range(len(index))]
        epi, weight, gamma_final = self.select_epi_rate(epi_sort, TI_sort, gamma_sort)
        return epi, weight, gamma_final

    def graph_construction(self, X, epi, weight, gamma_final):
        W_PD = np.zeros((len(X), len(X)))
        for i in range(len(epi)):
            gamma = 1 / (2 * np.power(gamma_final[i], 2))
            distance_matrix = ms.pairwise.rbf_kernel(X, gamma=gamma)
            connectivity = radius_neighbors_graph(
                X, epi[i], mode="connectivity", include_self=False
            )
            connectivity = np.einsum("ij,ij->ij", distance_matrix, connectivity.todense())
            w = 0.5 * (connectivity + np.einsum("ji", connectivity))
            W_PD += weight[i] * w
        return W_PD

    def fit_transform(self, X):
        H0 = self.computePH(X)
        survival_difference, para, epi_up = self.interval(H0)
        epi, weight, gamma = self.majorPara(survival_difference, para, epi_up)
        return self.graph_construction(X, epi, weight, gamma)
