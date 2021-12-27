from scipy.sparse import dok_matrix
import numpy as np

class DOKTensor(dok_matrix):
    def __init__(self, shape, joindims = None, dtype = None):
        self.__shape_nd = shape
        if joindims is None:
            n = len(shape)
            order = np.argsort(shape)
            order = np.concatenate([order[n//4:], order[:n//4]])
            joindims = np.full(n, False)
            joindims[order[:n//2]] = True
            print(f"Joining dims {joindims} for shape {shape}")

            self.__axis_order = order # doesn't need to be sorted by size: instead we can just send the indices where(joindims) to one side and the others to the other side
        self.__dimsN = joindims
        self.__dimsM = np.logical_not(joindims)

        N = np.multiply.reduce([n for n, join in zip(shape, self.__dimsN) if join])
        M = np.multiply.reduce([m for m, join in zip(shape, self.__dimsM) if join])
        print(f"Creating sparse array of size {N, M}")

        super().__init__((N, M), dtype=dtype)

    def __get_dim_index(self, index, joindims):
        shape = self.shape_nd
        assert len(index) == len(shape) == len(joindims), f"Dimensions do not match! {len(index)} == {len(shape)} == {len(joindims)}"
        # Filter out unrelated dims
        index = [i for i, join in zip(index, joindims) if join]
        shape = [n for n, join in zip(shape, joindims) if join]

        a = np.stack(index, axis=-1)
        b = np.cumprod([1] + shape[:-1])
        return np.sum(np.multiply(a, b), axis=-1) # I could use ravel_multi_index and unravel_index instead

    def __getitem__(self, index):
        index = np.broadcast_arrays(*index)
        i = self.__get_dim_index(index, self.__dimsN)
        j = self.__get_dim_index(index, self.__dimsM)
        i, j = i.flatten(), j.flatten()
        out = super().__getitem__((i, j))
        return out.toarray().reshape(index[0].shape)

    def __setitem__(self, index, value):
        index = np.broadcast_arrays(*index)
        i = self.__get_dim_index(index, self.__dimsN)
        j = self.__get_dim_index(index, self.__dimsM)
        i, j, value = i.flatten(), j.flatten(), value.flatten()
        return super().__setitem__((i, j), value)

    @property
    def shape_2d(self):
        return self.shape

    @property
    def shape_nd(self):
        return self.__shape_nd

    def arr_nd_to_2d(self, arr):
        n = len(self.shape_nd)
        arr = arr.moveaxis(self.__axis_order, np.arange(n))
        arr = arr.reshape(self.shape_2d)
        return arr

    def arr_2d_to_nd(self, arr):
        n = len(self.shape_nd)
        arr = arr.toarray()
        arr = arr.reshape(self.shape_nd)
        arr = arr.moveaxis(np.arange(n), self.__axis_order)
        return arr
