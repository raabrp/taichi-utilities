"""
Sort with bitonic merge sorting network
https://en.wikipedia.org/wiki/Bitonic_sorter

Provides:

@ti.func
def ti_sort(data):
    '''
    Sort data, in-place.

    `data` is a one-dimensional taichi object that supports
    comparision with `>`.

    calls ti_sort_serial   if length of data is <= 8
    calls ti_sort_parallel if length of data is > 8
    '''
                        ...

@ti.func
def ti_sort_duals(data, duals):
    '''
    Sort data in place, as in `ti_sort`.

    Also perform the same permuation on each array in `duals`.

    calls ti_sort_serial   if length of data is <= 8
    calls ti_sort_parallel if length of data is > 8
    '''
                        ...

@ti.func
def ti_sort_serial(data, duals):
    '''
    Runs in a single thread.
    O(n log n), but no runtime overhead computing indices.
    Use for performing multiple sorts on small N in parallel.
    '''
                        ...

@ti.func
def ti_sort_parallel(data, duals):
    '''
    Will parallelize.
    O(log n), but incurs runtime overhead for calculating network indices.
    Use for large N.
    '''
                        ...

"""

import math
import taichi as ti


################################################################################


@ti.func
def ti_sort(data: ti.template()):
    ti_sort_duals(data, ())


@ti.func
def ti_sort_duals(data: ti.template(), duals: ti.template()):
    if ti.static(ti_len(data)) <= 8:
        ti_sort_serial(data, duals)
    else:
        ti_sort_parallel(data, duals)


def ti_len(data):
    # assume any expression defined in ti scope
    # that we're trying to sort must be a ti.Vector
    if isinstance(data, tuple):
        return len(data)
    if isinstance(data, ti.lang.expr.Expr) or isinstance(data, ti.lang.Vector):
        return data.n
    else:
        return data.shape[0]


################################################################################


@ti.func
def ti_sort_serial(data: ti.template(), duals: ti.template()):
    """
    Runs in a single thread.
    O(n log n), but no runtime overhead computing indices.
    Use for performing multiple sorts on small N in parallel.
    """
    ti.loop_config(serialize=True)
    for i, j in ti.static(precompute_network(ti_len(data))):
        compare_swap(data, duals, i, j)


def precompute_network(N):
    # list of pairs to compare and swap if necessary
    comparisons = []

    l2N = math.log2(N)

    # number of merging stages to perform
    M_ceil = int(math.ceil(l2N))

    # with each merge, we assume that we are given two blocks of size 2^b, each
    # of which is already sorted.  We produce a block up to size 2^(b+1) that is
    # sorted.
    for b in range(M_ceil):
        # size of blocks that are already guaranteed to be sorted
        B = 1 << b  # [1, 2, 4, ..., 2^(M_ceil-1)]

        # The number of comparisions to make within the complete
        # blocks of size 2^(b+1) that fit within N
        H = (N >> (b + 1)) << b  # = B * (N // (2 * B))

        # the remaining comparisons to make within any remaining
        # partial blocks
        P = ((N >> b) & 1) * (N & (B - 1))  # = max(N - 2 * H - B, 0)

        # This for-loop can be performed in parallel. It corresponds to slice of
        # the network that looks like a stack of triangles:
        #
        # b=1      b=2        b=4
        # -I-     ---I-     -------I-
        # -J- ... -I-|- ... -----I-|- ...
        # -I-     -J-|-     ---I-|-|-
        # -J-     ---J-     -I-|-|-|-
        # -I-     ---I-     -J-|-|-|-
        # -J- ... -I-|- ... ---J-|-|- ...
        # -I-     -J-|-     -----J-|-
        # -J-     ---J-     -------J-
        #
        for k in range(H + P):
            # first index within this block of size 2*B
            i = (B - 1 - k) & (B - 1)

            # second index within this block of size 2*B
            j = (B << 1) - 1 - i

            # block offset
            bo = (k >> b) << (b + 1)

            comparisons += [(bo + i, bo + j)]

        # iterate over strides of length 2^s
        # this for-loop must be executed sequentially
        for p in range(b):
            s = b - p - 1

            # stride
            S = 1 << s  # [B/2, B/4, ..., 1]

            # The number of comparisions to make within the complete
            # blocks of size 2^(s+1) that fit within N
            H = (N >> (s + 1)) << s  # = B * (N // (2 * S))

            # the remaining comparisons to make within any remaining
            # partial blocks
            P = ((N >> s) & 1) * (N & (S - 1))  # = max(N - 2 * H - S, 0)

            # This for-loop can be performed in parallel. It corresponds to a
            # slice of the network that looks like a stack of rhombuses:
            #
            # s=1      s=2         s=3
            # -I-     -I---     -I-------
            # -J-     -|-I-     -|-I-----
            # -I- ... -J-|- ... -|-|-I--- ...
            # -J-     ---J-     -|-|-|-I-
            # -I-     -I---     -J-|-|-|-
            # -J- ... -|-I- ... ---J-|-|- ...
            # -I-     -J-|-     -----J-|-
            # -J-     ---J-     -------J-
            #
            for k in range(H + P):
                # first index within block
                i = k & (S - 1)

                # second index within block
                j = i + S

                # block offset
                # mod 2^s, then multiply by 2
                bo = (k >> s) << (s + 1)

                comparisons += [(bo + i, bo + j)]

    return comparisons


################################################################################


@ti.func
def ti_sort_parallel(data: ti.template(), duals: ti.template()):
    """
    Will parallelize.
    O(log n), but incurs runtime overhead for calculating network indices.
    Use for large N.
    """

    N = ti.static(ti_len(data))
    l2N = ti.static(math.log2(N))
    M_ceil = ti.static(int(math.ceil(l2N)))

    # this for-loop must be executed sequentially
    ti.loop_config(serialize=True)
    for b in range(M_ceil):
        B = 1 << b
        H = (N >> (b + 1)) << b
        P = ((N >> b) & 1) * (N & (B - 1))

        butterfly_triangle(data, duals, H + P, B, b)

        # this for-loop must be executed sequentially
        for p in range(b):
            s = b - p - 1
            S = 1 << s
            H = (N >> (s + 1)) << s
            P = ((N >> s) & 1) * (N & (S - 1))

            butterfly_rhombus(data, duals, H + P, S, s)


@ti.func
def butterfly_rhombus(
    data: ti.template(), duals: ti.template(), K: int, S: int, s: int
):
    # Perform in parallel
    for k in range(K):
        i = k & (S - 1)
        j = i + S
        bo = (k >> s) << (s + 1)
        compare_swap(data, duals, bo + i, bo + j)


@ti.func
def butterfly_triangle(
    data: ti.template(), duals: ti.template(), K: int, B: int, b: int
):
    # Perform in parallel
    for k in range(K):
        i = (B - 1 - k) & (B - 1)
        j = (B << 1) - 1 - i
        bo = (k >> b) << (b + 1)
        compare_swap(data, duals, bo + i, bo + j)


################################################################################


@ti.func
def compare_swap(data: ti.template(), duals: ti.template(), i: int, j: int):
    """
    Swap data entries at indices i and j so that
    data[i] < data[j] afterwards
    """
    if data[i] > data[j]:
        tmp = data[j]
        data[j] = data[i]
        data[i] = tmp

        for k in ti.static(range(ti_len(duals))):
            t_tmp = duals[k][j]
            duals[k][j] = duals[k][i]
            duals[k][i] = t_tmp


################################################################################


def viz_network(N):
    """
    Visualize generated network for debugging
    """
    lines = [f"{l}: " for l in range(N)]
    pairs = precompute_network(N)
    for i, j in pairs:
        for l in range(N):
            if isinstance(i, str):
                lines[l] += i
            elif l == i:
                lines[l] += "I"
            elif l == j:
                lines[l] += "J"
            elif i < l < j:
                lines[l] += "|"
            else:
                lines[l] += "-"
            lines[l] += "-"
    for p in lines:
        print(p)


################################################################################

# run as python3 -m taichi_utils.sort
if __name__ == "__main__":
    import unittest
    import numpy as np

    ti.init(
        arch=ti.gpu,
        default_ip=ti.i32,  # int
        default_fp=ti.f32,  # float
    )

    class Test(unittest.TestCase):
        """
        Test ti_sort by using it to sort random inputs
        """

        def test_network(self):
            max_length = 100
            num_tests_each_length = 10

            for N in range(1, max_length):
                pairs = precompute_network(N)

                for _ in range(num_tests_each_length):
                    data = list(np.random.random(N))
                    cpy = [x for x in data]

                    for i, j in pairs:
                        if isinstance(i, str):
                            continue
                        if data[i] > data[j]:
                            tmp = data[j]
                            data[j] = data[i]
                            data[i] = tmp

                    cpy.sort()

                    self.assertTrue(data == cpy)

        def test_ti_sort_field(self):
            max_length = 10
            num_tests_each_length = 10

            for N in range(1, max_length):
                data = ti.field(dtype=float, shape=((N,)))

                @ti.kernel
                def test_kernel():
                    ti_sort(data)

                for _ in range(num_tests_each_length):
                    # load randomly generated floats
                    np_data = np.random.random((N,))
                    data.from_numpy(np_data)

                    # do the sort
                    test_kernel()

                    # compare to numpy
                    x = np.sort(np_data)
                    ti_x = data.to_numpy()
                    self.assertTrue(np.allclose(x, ti_x), msg=f"{x}, {ti_x}")

        def test_ti_sort_duals(self):
            max_length = 10
            num_tests_each_length = 10

            for N in range(1, max_length):
                data = ti.field(dtype=float, shape=((N,)))
                dual = ti.field(dtype=int, shape=((N,)))

                @ti.kernel
                def test_kernel():
                    ti_sort_duals(data, (dual,))

                for _ in range(num_tests_each_length):
                    # load randomly generated floats
                    np_data = np.random.random((N,))
                    data.from_numpy(np_data)

                    # load indices
                    np_dual = np.arange(N)
                    dual.from_numpy(np_dual)

                    # do the sort
                    test_kernel()

                    # compare to numpy
                    x = np.argsort(np_data)
                    ti_x = dual.to_numpy()
                    self.assertTrue(np.allclose(x, ti_x), msg=f"{x}, {ti_x}")

        def test_ti_sort_vector(self):
            """
            Use sorting method on a ti.Vector input
            """
            max_length = 10
            num_tests_each_length = 10

            for N in range(1, max_length):
                pairs = precompute_network(N)

                data = ti.field(dtype=float, shape=((N,)))

                @ti.kernel
                def test_kernel(np_data_in: ti.types.ndarray()):
                    v = ti.Vector.zero(float, N)
                    for i in range(N):
                        v[i] = np_data_in[i]
                    ti_sort(v)
                    for i in range(N):
                        data[i] = v[i]

                for _ in range(num_tests_each_length):
                    # load randomly generated floats and do the sort
                    np_data = np.random.random((N,))
                    test_kernel(np_data)

                    # compare to numpy
                    x = np.sort(np_data)
                    ti_x = data.to_numpy()
                    self.assertTrue(np.allclose(x, ti_x), msg=f"{x}, {ti_x}")

    unittest.main()
