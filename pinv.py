"""
Solve for Moore-Penrose pseudo-inverse as fixed-point of (Gauss)-Newton
Iteration:

x_{ij}^{t+1} = 2 x_{ij}^{t} - x_{ik}^t A_{km} x_{mj}^t

Proof of convergence:
https://aalexan3.math.ncsu.edu/articles/mat-inv-rep.pdf
"""

import taichi as ti
import taichi.math as tm


@ti.func
def step(A: ti.template(), x: ti.template()) -> ti.template():
    return 2.0 * x - x @ A @ x


@ti.func
def pinv(A: ti.template(), n: int) -> ti.template():
    b = A @ A.transpose()
    bj = ti.Vector.zero(float, A.n)

    # in parallel
    for j in range(b.n):
        # serial
        for i in range(A.m):
            bj[j] += ti.abs(b[i, j])

    R = bj.max()
    x = A.transpose() / R

    # serial
    ti.loop_config(serialize=True)
    for _ in range(n):
        x = step(A, x)

    return x


# run as python3 -m taichi_utils.pinv
if __name__ == "__main__":
    ti.init(
        arch=ti.gpu,
        default_ip=ti.i32,  # int
        default_fp=ti.f32,  # float
    )

    out = ti.Matrix.field(dtype=float, n=2, m=2, shape=())

    @ti.kernel
    def test():
        A = ti.Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out[None] = pinv(A, 15) @ A

    test()

    print(out.to_numpy())
