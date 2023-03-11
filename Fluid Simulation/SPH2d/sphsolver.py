import taichi as ti
import numpy as np



@ti.data_oriented
class SPHSolver:
    def __init__(self, particle_system) -> None:
        self.particle_s = particle_system
        self.target_den = 1000.0 
        self.p_mass = self.particle_s.m_V * self.target_den
        #pressure equation of state
        self.eosscale = 50.0
        self.eosexponent = 7.0
        #none pressure force attribute
        self.gravity = ti.Vector([0, -9.8])
        self.viscosity = 0.05

        self.force = ti.Vector.field(2, ti.f32)
        force_node = ti.root.dense(ti.i, self.particle_s.particle_max_num)
        force_node.place(self.force)
        self.dt = 2e-3

        #collision
        self.collider = ti.Vector.field(2, ti.f32)
        
        self.collider_node = ti.root.dense(ti.i, 10)
        self.collider_node.place(self.collider)

        self.collider_num = ti.field(ti.i32, shape=())
        self.collider_num[None] = 0
    @ti.func
    def sph_kernel(self, dis):
        res = ti.cast(0.0, ti.f32)
        h = self.particle_s.neighbor_radius
        k = 40 / 7 / np.pi
        k/= h ** 2
        q = dis / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res
    
    @ti.func
    def sph_kernel_derivative1(self, r):
        h = self.particle_s.neighbor_radius
        k = 40 / 7 / np.pi
        k = 6. * k / h ** 2
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0, 0])
        if r_norm > 1e-5 and q <= 1:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def sph_kernel_derivative2(self, dis):
        pass

    @ti.func
    def sph_kernel_gradient(self, dis, dir):
        return -self.sph_kernel_derivative(dis) * dir
    
    @ti.kernel
    def update_den(self):
        num = self.particle_s.num_particle[None]
        for i in range(num):
            sum = 0
            self.particle_s.density_p[i] = 0.0
            for j in range(self.particle_s.particle_neighbors_num[i]):
                p_nei = self.particle_s.particle_neighbors[i, j]#第i个粒子的第j个邻居
                dis = (self.particle_s.positions_p[i] - self.particle_s.positions_p[p_nei]).norm()
                self.particle_s.density_p[i] += self.sph_kernel(dis) * self.particle_s.m_V
            self.particle_s.density_p[i] *= self.target_den

    @ti.func
    def interpolate(self, origin, values):
        sum = ti.Vector([0, 0])
        nei_index = self.particle_s.get_neighbors(origin)
        for i in range(len(nei_index)):
            dis = (self.particle_s.positions_p[nei_index[i]] - origin).norm()
            weight = self.p_mass / self.p_density[i] * self.sph_kernel(dis)
            sum += weight * values[nei_index[i]]
        return sum

    @ti.func
    def pos_gradient(self, i, values:ti.template()):
        p = self.particle_s.positions_p
        d = self.p_density
        sum = ti.Vector([0, 0])
        origin = p[i]
        neighbors_num = self.particle_s.particle_neighbors_num[i]
        for j in range(neighbors_num):
            nn = self.particle_s.particle_neighbors[i, j]
            neighbor_pos = p[nn]
            dis = (origin - neighbor_pos ).norm()
            if dis > 0.0:
                dir = () / dis
                sum += d[i] * self.p_mass * (values[i] / d[i]**2 + values[nn] / d[nn]**2) * self.sph_kernel_gradient(dis, dir)
        
        return sum

    @ti.func
    def pos_laplacian(self, i, value:ti.template()):
        p = self.particle_s.positions_p
        d = self.particle_s.density_p

        sum = 0.0
        origin = p[i]
        for j in range(self.particle_s.particle_neighbors_num[i]):
            nn = self.particle_s.particle_neighbors[i, j]
            neighbor_pos = p[nn]
            dis = (origin - neighbor_pos).norm()
            sum += self.p_mass * (value[nn] - value[i]) / d[nn] * self.sph_kernel_derivative2(dis)

        return sum

    @ti.kernel
    def compute_pressure_force(self):
        
        #compute pressure
        for p_i in range(self.particle_s.num_particle[None]):
            self.particle_s.pressure[p_i] = self.eosscale * (ti.pow((self.particle_s.density_p[p_i] / self.target_den), self.eosexponent) - 1.0)
        #compute force
        for p_i in range(self.particle_s.num_particle[None]):
            if self.particle_s.material[p_i] != 1:
                continue
            pos_i = self.particle_s.positions_p[p_i]
            force = ti.Vector([0, 0])
            for j in range(self.particle_s.particle_neighbors_num[p_i]):
                p_j = self.particle_s.particle_neighbors[p_i, j]
                pos_j = self.particle_s.positions_p[p_j]
                force += -self.target_den * self.particle_s.m_V * (self.particle_s.pressure[p_i] / self.particle_s.density_p[p_i]**2
                        + self.particle_s.pressure[p_j] / self.particle_s.density_p[p_j] ** 2) \
                        * self.sph_kernel_derivative1(pos_i - pos_j)
            self.force[p_i] += force
    
    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.particle_s.velocities_p[p_i] -
                self.particle_s.velocities_p[p_j]).dot(r)
        res = 2 * (4) * self.viscosity * (self.p_mass / (self.particle_s.density_p[p_j])) * v_xy / (
            r.norm()**2 + 0.01 * 0.2**2) * self.sph_kernel_derivative1(
                r)
        return res
    
    @ti.kernel
    def compute_viscosity_force(self):#together with gravity
        for p_i in range(self.particle_s.num_particle[None]):
            if self.particle_s.material[p_i] != 1:
                continue
            pos_i = self.particle_s.positions_p[p_i]
            force = ti.Vector([0, 0])
            #add gravity
            force = ti.Vector([0, -9.8])
            for j in range(self.particle_s.particle_neighbors_num[p_i]):
                p_j = self.particle_s.particle_neighbors[p_i, j]
                pos_j = self.particle_s.positions_p[p_j]
                #add viscosity force
                force += self.viscosity_force(p_i, p_j, pos_i - pos_j)
            self.force[p_i] = force
    
    @ti.kernel
    def advect_p(self):
        for p_i in range(self.particle_s.num_particle[None]):
            if self.particle_s.material[p_i] == 1:
                self.particle_s.velocities_p[p_i] += self.dt * self.force[p_i]
                self.particle_s.positions_p[p_i] += self.dt * self.particle_s.velocities_p[p_i]
    
    def add_collider(self, pos):
        if self.collider_num[None] < 10:
            self.collider[self.collider_num[None]] = pos
            self.collider_num[None] += 1

    @ti.func
    def handle_collision(self, p_i, vec, d):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.3
        self.particle_s.positions_p[p_i] += vec * d
        self.particle_s.velocities_p[p_i] -= (
            1.0 + c_f) * self.particle_s.velocities_p[p_i].dot(vec) * vec
    @ti.func
    def handle_collision2(self, p_i, vec, d):
        c_f = 0.3
        self.particle_s.positions_p[p_i] += vec * d / 10
        self.particle_s.velocities_p[p_i] -= (
            1.0 + c_f) * self.particle_s.velocities_p[p_i].dot(vec) * vec
    @ti.kernel
    def handle_boundary(self):
        for p_i in range(self.particle_s.num_particle[None]):
            if self.particle_s.material[p_i] == 1:
                pos = self.particle_s.positions_p[p_i]
                for j in range(self.collider_num[None]):
                    vec = (pos - self.collider[j])
                    if vec.norm()<=1:
                        self.handle_collision2(p_i,
                                               vec.normalized(),
                                               0.3)
                if pos[0] < 0.2:
                    self.handle_collision(p_i,
                        ti.Vector([1.0, 0]),
                        0.2 - pos[0])
                if pos[0] >  10.0:
                    self.handle_collision(p_i,
                        ti.Vector([-1.0, 0]),
                        pos[0] - 10)
                if pos[1] < 0.2:
                    self.handle_collision(p_i,
                        ti.Vector([0, 1]),
                        0.2 - pos[1])
                if pos[1] > 10.0:
                    self.handle_collision(p_i,
                        ti.Vector([0, -1]),
                        pos[1] - 10.0)
    
    @ti.kernel
    def copy_to_numpy_nd(self, np_arr: ti.ext_arr(), src_arr: ti.template()):
        for i in range(self.collider_num[None]):
            for j in ti.static(range(2)):
                np_arr[i, j] = src_arr[i][j]

    def dump(self):           
        np_x = np.ndarray((self.collider_num[None], 2), dtype=np.float32)
        self.copy_to_numpy_nd(np_x, self.collider)
        return np_x
    
    def substep(self):
        self.update_den()
        self.compute_viscosity_force()
        self.compute_pressure_force()
        self.advect_p()

    def step(self):
        self.particle_s.reset_particle_system()
        self.substep()
        self.handle_boundary()
