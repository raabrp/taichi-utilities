"""
Parallel Psuedo-Random Number Generators (e.g., for the GPU), where determinism
is important for recreatability. (Reusing the same seed should reproduce exactly
the same results.)

32-bit

The underlying method performed in parallel is xoshiro128+.

An initial random state is provided by numpy (PCG64).
"""

import numpy as np

import taichi as ti
import taichi.math as tm

uint = ti.u32


@ti.data_oriented
class RandomField:
    """
    Field of random generators that accepts a seed (int)
    and a shape (tuple of ints).

    Methods allow output as
    + uniform uint (unsigned 32 bit int)  ---  bits32
    + uniform k-bit value                 ---  bits
    + uniform float (32 bit), in [0, 1)   ---  random
    + uniform float (32 bit), in (-, 1)   ---  signed_random
    + gaussian vec2 (32 bit)              ---  randn

    Can be called per (grouped) index within given shape (methods above)
    or written directly to field from Python scope with methods
    + write_bits32,
    + write_bits,
    + write_random,
    + write_signed_random,
    + write_randn
    """

    def __init__(self, seed, shape):
        self.seed = seed
        self.shape = shape

        # declare internal state
        self.state = ti.Vector.field(n=4, dtype=uint, shape=shape)

        # use numpy to populate state with random values
        np_prng = np.random.default_rng(seed)  # PCG64
        random_np_array = np_prng.integers(
            low=0, high=2**32, size=(shape + (4,)), dtype=np.uint32
        )
        self.state.from_numpy(random_np_array)

    @staticmethod
    @ti.func
    def uint_to_unit_float(x: uint) -> float:
        """
        IEEE 754 32 bit unsigned int -> float in [0, 1)
        """
        exp = 127 << 23
        mantissa = x >> 9
        return ti.bit_cast(exp | mantissa, float) - 1.0

    @staticmethod
    @ti.func
    def uint_to_signed_float(x: uint) -> float:
        """
        IEEE 754 32 bit unsigned int -> float in (-1, 1)
        """
        exp = 127 << 23
        mantissa = (x << 1) >> 9
        f = ti.bit_cast(exp | mantissa, float) - 1.0
        if x >> 31:  # sign
            f = -f
        return f

    @staticmethod
    @ti.func
    def box_muller(v: tm.vec2) -> tm.vec2:
        """
        vec2 in [0, 1) -> vec2 of standard normal values
        """
        r = tm.sqrt(-2 * tm.log(v.x))
        t = 2 * tm.pi * v.y
        c = tm.cos(t)
        s = tm.sin(t)
        return tm.vec2(r * c, r * s)

    ############################################################################

    @ti.func
    def step(self, i_tup: ti.template()):
        """
        Update state of PRNG to next value
        """
        s = self.state[i_tup]

        self.state[i_tup][2] ^= s[0]
        self.state[i_tup][3] ^= s[1]
        self.state[i_tup][1] ^= s[2]
        self.state[i_tup][0] ^= s[3]

        self.state[i_tup][2] ^= s[1] << 9

        self.state[i_tup][3] = (s[3] << 11) | (s[3] >> 21)

    ############################################################################

    @ti.func
    def bits32(self, i_tup: ti.template()) -> uint:
        """
        Generate a new 32 bit unsigned int
        """
        self.step(i_tup)
        s = self.state[i_tup]
        return s[0] + s[3]

    @ti.func
    def bits(self, i_tup: ti.template(), k: int) -> uint:
        return self.bits32(i_tup) >> (32 - k)

    @ti.func
    def random(self, i_tup: ti.template()) -> float:
        """
        Generate a new uniform random value in [0, 1) (23 bits of entropy)
        """
        return self.uint_to_unit_float(self.bits32(i_tup))

    @ti.func
    def signed_random(self, i_tup: ti.template()) -> float:
        """
        Generate a new uniform random value in (-1, 1) (24 bits of entropy)
        """
        return self.uint_to_signed_float(self.bits32(i_tup))

    @ti.func
    def randn2(self, i_tup: ti.template()) -> tm.vec2:
        """
        Generate two new normally distributed values (as vec2)
        """
        v = tm.vec2(self.random(i_tup), self.random(i_tup))
        return self.box_muller(v)

    ############################################################################

    @ti.func
    def write_bits32(self, dst: ti.template()):
        """
        Write uniform unsigned int (32 bits) to field with matching shape
        """
        for i_tup in ti.grouped(self.state):
            dst[i_tup] = self.bits32(i_tup)

    @ti.func
    def write_bits(self, dst: ti.template(), k: int):
        """
        Write uniform unsigned int (32 bits) to field with matching shape
        """
        for i_tup in ti.grouped(self.state):
            dst[i_tup] = self.bits(i_tup, k)

    @ti.func
    def write_random(self, dst: ti.template()):
        """
        Write uniform random values in [0, 1) to field with matching shape
        (type float)
        """
        for i_tup in ti.grouped(self.state):
            dst[i_tup] = self.random(i_tup)

    @ti.func
    def write_signed_random(self, dst: ti.template()):
        """
        Write uniform random values in (-1, 1) to field with matching shape
        (type float)
        """
        for i_tup in ti.grouped(self.state):
            dst[i_tup] = self.signed_random(i_tup)

    @ti.func
    def write_randn2(self, dst: ti.template()):
        """
        Write standard normal values to field with matching shape
        (type vec2)
        """
        for i_tup in ti.grouped(self.state):
            dst[i_tup] = self.randn2(i_tup)


################################################################################

# run as python3 -m taichi_utils.prng
if __name__ == "__main__":
    import unittest
    import numpy as np
    from matplotlib import pyplot as plt

    ti.init(
        arch=ti.cpu,
        default_ip=ti.i32,  # int
        default_fp=ti.f32,  # float
    )

    shape = (4096,)
    N = 1
    rng = RandomField(seed=0, shape=shape)

    class Test(unittest.TestCase):
        def test_random(self):
            data = ti.field(dtype=float, shape=shape)

            @ti.kernel
            def test_kernel():
                rng.write_random(data)

            def show():
                test_kernel()
                plt.hist(data.to_numpy().flatten(), density=True)
                plt.show()

            for _ in range(N):
                show()

        def test_signed_random(self):
            data = ti.field(dtype=float, shape=shape)

            @ti.kernel
            def test_kernel():
                rng.write_signed_random(data)

            def show():
                test_kernel()
                plt.hist(data.to_numpy().flatten(), density=True)
                plt.show()

            for _ in range(N):
                show()

        def test_randn(self):
            data = ti.Vector.field(n=2, dtype=float, shape=shape)

            @ti.kernel
            def test_kernel():
                rng.write_randn2(data)

            def show():
                test_kernel()
                plt.hist(data.to_numpy().flatten(), density=True)
                plt.show()

            for _ in range(N):
                show()

    unittest.main()
