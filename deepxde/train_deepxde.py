from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import deepxde as dde
from deepxde.backend import tf

def main():
    # 定义模型的参数
    omiga = 3.497e-6
    nu = 0.3
    D = 7.08e-15
    E = 1e10
    R1 = 8e-7
    j = 1e-3
    R = 8.3145
    T = 300
    R2 = 4e-7
    alfa = j * R1 * omiga / D
    theta = omiga * E / (R * T * 3.0 * 0.4)

    # 定义PDE
    def pde(x, y):
        u, c = y[:, 0:1], y[:, 1:]
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        dc_x = dde.grad.jacobian(y, x, i=1, j=0)
        dc_t = dde.grad.jacobian(y, x, i=1, j=1)
        # du_xx = dde.grad.hessian(y, x, i=0, j=0)
        # dc_xx = dde.grad.hessian(y, x, i=1, j=0)
        du_xx = tf.gradients(du_x, x)[0][:, 0:1]
        dc_xx = tf.gradients(dc_x, x)[0][:, 0:1]
        du_xxx = tf.gradients(du_xx, x)[0][:, 0:1]
        stress_equation = x[:, 0:1] ** 2 * du_xx + x[:, 0:1] * du_x - u - x[:, 0:1] ** 2 * dc_x * alfa * 1.3 / (
                    0.7 * 3.0)
        # 3.0*(1-nu)*F*D*du_x-2.0*nu*omiga*I*H*c
        diffusion_equation = x[:, 0:1] ** 3 * dc_t - x[:, 0:1] ** 3 * dc_xx - x[:, 0:1] ** 2 * dc_x
        return [stress_equation, diffusion_equation]

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0.5)

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

    '''def stress_bc(x, y, X):
        u, c = y[:, 0:1], y[:, 1:]
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        alfa=(1.0+du_x)
        beta=(1.0+omiga*I*H*c/(F*D))
        return nu/(1.0-2*nu)*((beta)**(-2.0/3.0)-1.0+0.5*((alfa)**2*beta**(-2.0/3.0)-1.0))+0.5*(beta**(-2.0/3.0)-1.0)'''

    def stress_bc(x, y, X):
        u, c = y[:, 0:1], y[:, 1:]
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        return 0.7 / (1.3) * du_x * x[:, 0:1] + 0.3 / (1.3) * u - alfa * c * x[:, 0:1] / (3.0)

    def diffusion_bc_l(x, y, X):
        u, c = y[:, 0:1], y[:, 1:]
        dc_x = dde.grad.jacobian(y, x, i=1, j=0)
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_xx = tf.gradients(du_x, x)[0][:, 0:1]
        return dc_x

    def diffusion_bc_r(x, y, X):
        u, c = y[:, 0:1], y[:, 1:]
        dc_x = dde.grad.jacobian(y, x, i=1, j=0)
        du_x = dde.grad.jacobian(y, x, i=0, j=0)
        du_xx = tf.gradients(du_x, x)[0][:, 0:1]
        return dc_x - 1.0

        # 定义几何区域和时间域
    geom = dde.geometry.Interval(0.5, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_l_stress = dde.OperatorBC(geomtime, stress_bc, boundary_l)
    bc_r_stress = dde.OperatorBC(geomtime, stress_bc, boundary_r)
    #bc_l_diffusion = dde.NeumannBC(geomtime, lambda x: 0, boundary_l, component=1)
    bc_l_diffusion = dde.OperatorBC(geomtime, diffusion_bc_l, boundary_l)
    bc_r_diffusion = dde.OperatorBC(geomtime, diffusion_bc_r, boundary_r)

    # 定义初始条件
    ic_stress = dde.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=0)
    ic_diffusion = dde.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=1)

    # 定义数据
    data = dde.data.TimePDE(
        geomtime, pde, [bc_l_stress, bc_r_stress, bc_l_diffusion, bc_r_diffusion, ic_stress, ic_diffusion],
        num_domain=5000, num_boundary=500, num_initial=200
    )

    # 定义网络结构
    net = dde.maps.FNN([2] + [32] * 3 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # 编译模型
    model.compile("adam", lr=1e-4)
    # 训练模型
    model.train(epochs=10000)

    # 使用L-BFGS算法进行更精细的优化
    model.compile("L-BFGS-B")
    losshistory, train_state = model.train()

    # 保存训练好的模型
    model.save("model/model.ckpt")


if __name__ == "__main__":
    main()
