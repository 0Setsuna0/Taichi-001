import taichi as ti
ti.init(arch = ti.gpu)

num_particle = 8100
num_grid = 128
dx = 1 / num_grid
inv_dx = num_grid
dt = 1e-4

p_density = 1
p_volume = (dx * 0.5)**2
p_mass = p_volume * p_density
E = 0.1e4 # Young's modulus
nu = 0.2 # Poisson's ratio
mu0, lambda0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu)) # Lame parameters in Neo-Hookean model
x = ti.Vector.field(2, dtype = ti.f32, shape = num_particle)
v = ti.Vector.field(2, dtype = ti.f32, shape = num_particle)
C = ti.Matrix.field(2, 2, dtype = ti.f32, shape = num_particle)
F = ti.Matrix.field(2, 2, dtype = ti.f32, shape = num_particle)
J = ti.field(dtype = ti.f32, shape = num_particle) # defomation gradient determinate

grid_v = ti.Vector.field(2, dtype = ti.f32, shape = (num_grid, num_grid))
grid_m = ti.field(dtype = ti.f32, shape = (num_grid, num_grid))

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x: # P2G
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        #Quadratic B-Sline
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        #deformation update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p] # see in equation 162
        #handle snow material hardening
        h = ti.exp(10 * (1.0 - J[p]))
        #apply hardening parameter,see in euqation 87
        mu = mu0 * h
        la = lambda0 * h
        #svd, apply deformation to deformation gradient
        U, sig, V = ti.svd(F[p])
        J2 = 1.0
        for d in ti.static(range(2)):#clamp, hand diagonal element
            new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2),
                                 1 + 4.5e-3)  # Plasticity
            J[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J2 *= new_sig #use J2 to denote the determinant of sig,see in equation 50
        F[p] = U @ sig @ V.transpose()
        # calculate stress,see in equation 52
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J2 * (J2 - 1)
        stress = (-dt * p_volume * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
        #apply force and handle boundary condition
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
            grid_v[i, j][1] -= dt * 50 #gravity
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > num_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > num_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x: #G2P
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]

group_size = num_particle // 1
@ti.kernel
def initialize():
    for i in range(num_particle):
        x[i] = [
            ti.random() * 0.4 + 0.3 + 0.3 * (i // group_size),
            ti.random() * 0.4 + 0.05 + 0.42 * (i // group_size)
        ]
        v[i] = ti.Vector([0, 0])
        F[i] = ti.Matrix([[1,0], [0, 1]])
        J[i] = 1
def main():
    initialize()
    gui = ti.GUI("mpm snow",res=512, background_color = 0x112F41)
    e = gui.get_events(ti.GUI.PRESS)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == 'R':
               initialize()
               print(1)
        for s in range(int(2e-3 // dt)):
            substep()
        gui.circles(x.to_numpy(), radius=1.5, color=0xffffff)
        gui.show()

if __name__ == '__main__':
    main()
