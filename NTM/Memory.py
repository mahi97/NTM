import numpy as np


class Memory:
    def __init__(self, N, M):
        self.M = M
        self.N = N
        self.mem = np.zeros((N, M))

    def read(self, address):
        """

        :param address: NP-Array with Size of N, contain value between 0 and 1 with sum equals to 1
        :return: NP-Array with Size of M, produce by sum over weighted elements of Memory
        """
        return np.sum((address * self.mem.T).T, axis=0)

    def write(self, address, erase_vector, add_vector):
        self.erase(address, erase_vector)
        self.add(address, add_vector)

    def erase(self, address, erase_vector):
        self.mem = np.array([self.mem[i]*(np.ones(self.M) - address[i]*erase_vector) for i in range(self.N)])

    def add(self, address, add_vector):
        self.mem = np.array([self.mem[i] + address[i]*add_vector for i in range(self.N)])
