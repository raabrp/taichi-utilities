"""
Cooley-Tukey Fast Fourier Transform (FFT) for power-of-2-length input.
Breadth-first, decimation in time (DIT), radix-2, in-place
Implemented for GPU with Taichi.
32-bit precision.
"""

import taichi as ti
import taichi.math as tm

from twiddle import reverse_bits, int_log2

uint = ti.u32


@ti.func
def fft(data: ti.template(), inv: bool):
    """
    Multidimensional C2C Cooley-Tukey on power-of-2-length input.
    FFT performed on all axes.
    (Called from Taichi scope)

    data:      The data to transform, as a complex-valued ti.field (n=2)
               Can be multiple dimensions.
    inv:       Whether to take the inverse transformation.
    """

    # FFT in each dimension. Cannot be parallelized.
    # Uses compile-time forced loop unrolling
    for a in ti.static(range(len(data.shape))):
        fft_1D(data, a, inv)


@ti.func
def fft_on_axes(data: ti.template(), axes: ti.template(), inv: bool):
    """
    Multidimensional C2C Cooley-Tukey on power-of-2-length input.
    FFT performed on selected axes.
    (Called from Taichi scope)

    data:      The data to transform, as a complex-valued ti.field (n=2)
               Can be multiple dimensions.
    axes:      The index of the dimension to take the FFT over.
    inv:       Whether to take the inverse transformation.
    """

    # FFT in each dimension. Cannot be parallelized.
    # Uses compile-time forced loop unrolling
    for a in ti.static(range(len(axes))):
        fft_1D(data, axes[a], inv)


@ti.func
def fft_1D(data: ti.template(), axis: int, inv: bool):
    """
    Vectorized C2C Cooley-Tukey on power-of-2-length input.
    (Called from Taichi scope)

    data:  The data to transform, as a complex-valued ti.field (n=2)
           Can be multiple dimensions, but only one is transformed.
    axis:  The index of the dimension over which to take the FFT.
    inv:   Whether to perform the inverse transformation.
    """

    # bit-reversal permutation
    _permute_reverse_bits(data, axis)

    # By conjugate symmetries, we only need to iterate over half of the indices
    # in the FFT dimension. Precompute the shape to iterate over during the
    # butterfly layers here.
    vec_shape = ti.Vector(data.shape)  # convert from tuple to Taichi vector
    vec_shift = ti.zero(vec_shape)
    vec_shift[axis] = 1  # one-hot encode axis
    pshape = vec_shape >> vec_shift  # divides only data.shape[axis] by 2

    # recursive FFTs. Cannot be parallelized.
    ti.loop_config(serialize=True)
    for s in range(int_log2(vec_shape[axis])):
        _apply_butterfly_layer(data, axis, pshape, s, inv)

    # normalize (multiply by 1/N if inverse)
    _normalize(data, vec_shape[axis], inv)


@ti.func
def _normalize(data: ti.template(), N: int, inv: bool):
    """
    Perform normalization. While one could divide by sqrt(N) for both forward
    and backwards transformations, as is the convention in physics, this is not
    the standard convention for the FFT.
    """
    for idx in ti.grouped(data):
        if inv:
            data[idx] /= N


@ti.func
def _apply_butterfly_layer(
    data: ti.template(), axis: int, pshape: ti.template(), s: int, inv: bool
):
    """
    A single layer of the Cooley-Tukey FFT, after the bit-reversal permutation.
    With each call, we combine the results of 2 * W FFTs of size M/2 into the results
    of W FFTs of size M, in parallel, in-place.

    data:          data to transform
    axis:          which dimension to transform
    pshape:        shape that transfomation can be parallelized over. See _fft.
    s:             log2(stride) --- acts as layer index
    inv:           whether to perform inverse FFT
    """

    # M is size of FFTs we are currently performing with this call (determined
    # by s)
    M = 1 << (s + 1)  # is power of 2: s in [2, 4, 8, ..., 2^final_bit_width].
    # half_M is the size of the FFTs we are combining.
    half_M = 1 << s

    # in parallel
    for i_vec in ti.grouped(ti.ndrange(*pshape)):
        # The index that matters for performing the FFT is i.
        # All data entries with same index value i follow the same computational
        # path, in parallel.
        i = i_vec[axis]

        # Frequency index within each FFT.
        k = i & (half_M - 1)  # (i mod m/2).

        # Index for which FFT of size M we're addressing.
        j = i >> s  # (i // m/2).

        # Indices for places in data (along axis) to mutate.
        # These indices are used for both input and output, which allows us to
        # mutate the data in-place. This is the key difference between Cooley-Tukey
        # with a bit-reversal permutation up front and, e.g., Stockham (which
        # could be done in-place in theory, but fails in practice because the GPU
        # only has finitely many threads).
        ev = j * M + k
        od = ev + half_M

        # complex exponentiation: (Mth root of unity)^k
        twiddle = tm.vec2(tm.cos(tm.pi * k / half_M), -tm.sin(tm.pi * k / half_M))

        # If performing inverse FFT, need to take complex conjugate
        if inv:
            twiddle.y = -twiddle.y

        # restore non-FFT dimensions to vector indices.
        ev_vec = i_vec
        ev_vec[axis] = ev
        od_vec = i_vec
        od_vec[axis] = od

        # read values
        ev_val = data[ev_vec]
        od_twiddled = tm.cmul(twiddle, data[od_vec])

        # write values
        data[ev_vec] = ev_val + od_twiddled
        data[od_vec] = ev_val - od_twiddled


@ti.func
def _permute_reverse_bits(data: ti.template(), axis: int):
    """
    Permute elements in data array in-place
    along given axis
    by bit-reversing the index in the given axis.
    """

    # How many bits in the axis index?
    N = int_log2(ti.Vector(data.shape)[axis])

    for idx in ti.grouped(data):
        i = idx[axis]  # get index along given axis
        _i = reverse_bits(i, N)  # index with bits reversed

        # reindex to account for other dimensions of data array
        _idx = idx  # copy current index
        _idx[axis] = _i  # overwrite index along given axis

        # Disambiguate to only perform each swap once
        if i <= _i:
            tmp = data[_idx]
            data[_idx] = data[idx]
            data[idx] = tmp


################################################################################

if __name__ == "__main__":
    import unittest
    import numpy as np

    ti.init(
        arch=ti.cpu,
        default_ip=ti.i32,  # int
        default_fp=ti.f32,  # float
    )

    # entry point from Python scope
    @ti.kernel
    def fft_kernel(data: ti.template(), inv: bool):
        fft(data, inv)

    size = 4
    np_prng = np.random.default_rng(0)

    def load_np_complex_as_vec2(ti_field, np_array):
        assert ti_field.shape == np_array.shape
        ti_shape = np_array.shape + (2,)
        re = np.real(np_array)
        im = np.imag(np_array)
        ti_field.from_numpy(np.dstack([re, im]).reshape(ti_shape))

    def load_ti_vec2_as_np_complex(ti_field):
        ti_array = ti_field.to_numpy()
        re = np.take(ti_array, 0, -1)
        im = np.take(ti_array, 1, -1)
        return re + (0 + 1j) * im

    class Test(unittest.TestCase):
        """
        Test fft against numpy implementation on random inputs
        """

        def test_fft(self):
            for num_dimensions in [1, 2]:
                shape = (size,) * num_dimensions
                ti_field = ti.Vector.field(n=2, dtype=float, shape=shape)

                for _ in range(30):
                    # generate random input with numpy
                    re = np_prng.random(shape)
                    im = np_prng.random(shape)
                    x = re + (0 + 1j) * im

                    # load into taichi field
                    load_np_complex_as_vec2(ti_field, x)

                    # ti_fft
                    fft_kernel(ti_field, inv=False)
                    ti_X = load_ti_vec2_as_np_complex(ti_field)

                    # np_fft
                    X = x
                    for d in range(num_dimensions):
                        X = np.fft.fft(X, axis=d)

                    self.assertTrue(np.allclose(X, ti_X), msg=f"{X}, {ti_X}")

                    # ti_inverse fft
                    fft_kernel(ti_field, inv=True)
                    ti_x = load_ti_vec2_as_np_complex(ti_field)

                    self.assertTrue(np.allclose(x, ti_x), msg=f"{x}, {ti_x}")

    unittest.main()
