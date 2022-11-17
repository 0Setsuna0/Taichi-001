import taichi as ti
ti.init()
n=ti.field(dtype=ti.i32,shape=())
n=128
gridlen=1/n
X=ti.Vector.field(3,dtype=ti.f32,shape=(n,n))
V=ti.Vector.field(3,dtype=ti.f32,shape=(n,n))
rkx=ti.Vector.field(3,dtype=ti.f32,shape=(n,n))
rkv1=ti.Vector.field(3,dtype=ti.f32,shape=(n,n))
rkv2=ti.Vector.field(3,dtype=ti.f32,shape=(n,n))
rkv3=ti.Vector.field(3,dtype=ti.f32,shape=(n,n))
rkv4=ti.Vector.field(3,dtype=ti.f32,shape=(n,n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)


rest_length=[]
for i in range(-1,2):#-1,0,1
    for j in range(-1,2):#-1,0,1
        if(i,j)!=(0,0):
            rest_length.append(ti.Vector([i,j]))

ball_radius=0.3
ball_center=ti.Vector.field(3,dtype=float,shape=(1,))
ball_center[0]=ti.Vector([0,0,0])

deltat=4e-2 /n
substeps = int(1 / 60 // deltat)
damping=0.99
stiffness=ti.field(dtype=ti.f32,shape=())
stiffness[None]=3e4
gravity=ti.Vector([0,-9.8,0])

@ti.kernel
def initialize_mass_spring():
    random=ti.Vector([ti.random()-0.5,ti.random()-0.5])*0.1
    for i,j in X:
        X[i,j]=[i*gridlen-0.5+random[0],0.6,j*gridlen-0.5+random[0]]
        rkx[i,j]=[i*gridlen-0.5+random[0],0.6,j*gridlen-0.5+random[0]]
        V[i,j]=[0,0,0]

@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0., 0.5, 1)
        else:
            colors[i * n + j] = (1, 0.5, 0.)


initialize_mesh_indices()

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = X[i, j]

@ti.kernel
def handle_collison(x:ti.template(),v:ti.template()):
    for I in ti.grouped(x):
        offset_to_center = x[I] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            normal = offset_to_center.normalized()
            v[I] -= min(v[I].dot(normal), 0) * normal

        
@ti.kernel
def rkstep(x:ti.template(),v:ti.template(),dt:ti.template()):
    for i in ti.grouped(v):
        v[i] += gravity * dt *2

    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(rest_length):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = gridlen * float(i - j).norm()
                # Spring force
                force += -stiffness[None] * d * (current_dist / original_dist - 1)

        v[i] += force * dt * 2

@ti.kernel
def updaterkx(v:ti.template(),dt:ti.template()):
    for I in ti.grouped(X):
        offset_to_center=rkx[I]-ball_center[0]
        if offset_to_center.norm()<=ball_radius:
            normal = offset_to_center.normalized()
            v[I] -= min(v[I].dot(normal), 0) * normal
        rkx[I]=X[I]+v[I]*dt
@ti.kernel
def update_rkvalue():
    for I in ti.grouped(V):
        rkv1[I]=V[I]
        rkx[I]=X[I]
@ti.kernel
def update_XV():
    for I in ti.grouped(X):
        X[I]+=deltat*(rkv1[I]+2*rkv2[I]+2*rkv3[I]+rkv4[I])/6
        V[I]=(rkv1[I]+2*rkv2[I]+2*rkv3[I]+rkv4[I])/6
@ti.kernel
def mid_update():
    for I in ti.grouped(X):
        X[I]+=deltat*rkv1[I]
        V[I]=rkv1[I]
def substep():
    update_rkvalue()
    rkstep(rkx,rkv1,0.5*deltat)
    updaterkx(rkv1,deltat)
    #update_XV()
    mid_update()
    handle_collison(X,V)   


window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

current_t = 0.0
initialize_mass_spring()

while window.running:
    if current_t > 1.5:
        # Reset
        initialize_mass_spring()
        current_t = 0

    for i in range(substeps):
        substep()
        current_t += deltat
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.7, 0, 0))
    canvas.scene(scene)
    window.show()
