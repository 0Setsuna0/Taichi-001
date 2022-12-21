import taichi as ti
ti.init(arch = ti.cuda)

num_particles = 8192
num_grid = 128
dx = 1 / num_grid
inv_dx = 1 / dx
dt = 2e-4

p_density = 1
p_volume = (dx * 0.5)**2
p_mass = p_volume * p_density
gravity = 9.8
G = ti.Vector.field(2, dtype=ti.f32 , shape=())
G[None] = ti.Vector([0, -1])
bound = 3
E = 400

x = ti.Vector.field(2, dtype = ti.f32, shape = (num_particles))
v = ti.Vector.field(2, dtype = ti.f32, shape = (num_particles))
C = ti.Matrix.field(2, 2, dtype = ti.f32, shape = (num_particles))
J = ti.field(dtype = ti.f32, shape = (num_particles))

grid_v = ti.Vector.field(2, dtype = ti.f32, shape = (num_grid, num_grid))
grid_m = ti.field(dtype = ti.f32, shape = (num_grid, num_grid))

@ti.kernel
def substep():
    #reset grid quantity
    for i, j in grid_m:
        grid_m[i, j] = 0
        grid_v[i, j] = ti.Vector([0,0])
    #particle to grid
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)#left-bottom grid point index
        fx = Xp - base#offset between particle and left-bottom grid point
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_volume * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]#highly based on APIC,addtionally put stress together with affine
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    # #normalize quantity(veclocity) and apply appulse to add gravity and meanwhile apply boundary condition
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j] += G[None] * gravity * dt
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > num_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > num_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    #grid to particle
    for i in x:
        base = int(x[i] / dx - 0.5)
        fx = x[i] / dx -base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Vector.zero(float, 2, 2)
        for i,j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[i] = new_v
        x[i] += dt * v[i]
        J[i] *= 1 + dt * new_C.trace()
        C[i] = new_C

@ti.kernel
def init():
    for i in range(num_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1
 
init()
gui = ti.GUI("mpm")
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        if e.key == ti.GUI.LMB:
            G[None] = (ti.Vector([e.pos[0], e.pos[1]]) - ti.Vector([0.5, 0.5])).normalized()
    for step in range(50):
        substep()
    gui.clear(0x112F42)
    gui.circles(x.to_numpy(), radius=1.4, color=0x068587)
    gui.show()
