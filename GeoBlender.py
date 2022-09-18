import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu,device_memory_GB=4)
vec3 = ti.types.vector(3,ti.f32)

nba_pos = np.load('b_pos.npy')
nba_radi = np.load('b_radi.npy')
nba_pos.astype(np.float32)
nba_radi.astype(np.float32)
r_max = np.max(nba_radi)
number = nba_pos.shape[0]
dimension = nba_pos.shape[1]

fixb_pos = np.load('fixb_pos.npy')
fixb_radi = np.load('fixb_radi.npy')
fixb_pos -= np.array([0,0,0.2])
fixb_pos.astype(np.float32)
fixb_radi.astype(np.float32)
fix_r_max = np.max(fixb_radi)
fix_n = fixb_pos.shape[0]

total_number = number + fix_n

density = 100
gkn = 8e3
restitution_coef = 0.001
gravity = -9.81
dt = 0.00005
substeps = 10
grid_n = 60
grid_size = 0.6 / grid_n

x_bound = 0.15
y_bound = 0.15
z_bound = 0.3

assert r_max*2 < grid_size

gts = ti.field(ti.i32,shape=())
gts[None]=0

@ti.dataclass
class ball:
    # Ball Attribute
    p: vec3
    prep: vec3
    m: ti.f32
    r: ti.f32
    v: vec3
    a: vec3
    f: vec3
    # Ball Property
    kn: ti.f32
    ks: ti.f32

bf = ball.field(shape=number)
fix_bf = ball.field(shape=fix_n)
tot_bf = ball.field(shape=total_number)

@ti.kernel
def init(pos:ti.types.ndarray(),radi:ti.types.ndarray(),pos2:ti.types.ndarray(),radi2:ti.types.ndarray()):
    for i in bf:
        for j in ti.static(range(dimension)):
            bf[i].p[j] = pos[i,j]
            bf[i].prep[j] = pos[i,j]
        bf[i].r = radi[i]
        bf[i].m = density * math.pi * (bf[i].r**3)*4/3
        bf[i].kn = gkn

        tot_bf[i] = bf[i]

    for i in fix_bf:
        for j in ti.static(range(dimension)):
            fix_bf[i].p[j] = pos2[i,j]
            fix_bf[i].prep[j] = pos2[i,j]
        fix_bf[i].r = radi2[i]
        fix_bf[i].m = density * math.pi * (fix_bf[i].r**3)*4/3
        fix_bf[i].kn = gkn

        tot_bf[i+number] = fix_bf[i]


init(nba_pos,nba_radi,fixb_pos,fixb_radi)


@ti.kernel
def update():
    for i in bf:
        a = bf[i].f / bf[i].m
        if gts[None]==0:
            bf[i].p += bf[i].v * dt + 0.5 * a * dt ** 2
            bf[i].a = a
        else:
            bf[i].v += (bf[i].a + a) * dt / 2.0
            bf[i].prep = bf[i].p
            bf[i].p += bf[i].v * dt + 0.5 * a * dt**2
            bf[i].a = a
        bf[i].v *= 0.9995

        tot_bf[i] = bf[i]
    gts[None] += 1

@ti.kernel
def apply_bc():
    bounce_coef = 0.3
    for i in bf:
        x = bf[i].p[0]
        y = bf[i].p[1]
        z = bf[i].p[2]

        if y - bf[i].r < -y_bound:
            bf[i].p[1] = -y_bound + bf[i].r
            bf[i].v[1] *= -bounce_coef

        elif y + bf[i].r > y_bound:
            bf[i].p[1] = y_bound - bf[i].r
            bf[i].v[1] *= -bounce_coef

        if x - bf[i].r < -x_bound:
            bf[i].p[0] = -x_bound + bf[i].r
            bf[i].v[0] *= -bounce_coef

        elif x + bf[i].r > x_bound:
            bf[i].p[0] = x_bound - bf[i].r
            bf[i].v[0] *= -bounce_coef

        if z - bf[i].r < -z_bound:
            bf[i].p[2] = -z_bound + bf[i].r
            bf[i].v[2] *= -bounce_coef

        elif z + bf[i].r > z_bound:
            bf[i].p[2] = z_bound - bf[i].r
            bf[i].v[2] *= -bounce_coef

        tot_bf[i] = bf[i]

@ti.func
def resolve(i, j):
    rel_pos = tot_bf[j].p - tot_bf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
    delta = -dist + tot_bf[i].r + tot_bf[j].r  # delta = d - 2 * r
    if delta > 0:
        normal = rel_pos / dist
        lct_kn = tot_bf[i].kn*tot_bf[j].kn/(tot_bf[i].kn+tot_bf[j].kn)
        f_n = normal*delta*lct_kn
        tot_bf[i].f += -f_n
        tot_bf[j].f -= -f_n

total_grid_n = grid_n*grid_n*grid_n
list_head = ti.field(dtype=ti.i32, shape=total_grid_n)
list_cur = ti.field(dtype=ti.i32, shape=total_grid_n)
list_tail = ti.field(dtype=ti.i32, shape=total_grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n,grid_n,grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=(grid_n,grid_n), name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n,grid_n,grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=total_number, name="particle_id")


@ti.kernel
def contact(tbf: ti.template()):
    for i in tbf:
        tbf[i].f = vec3(0.,0., gravity * tbf[i].m)

    grain_count.fill(0)

    for i in range(total_number):
        grid_idx = ti.floor((tbf[i].p+0.3)/grid_size, int)
        grain_count[grid_idx] += 1

    for i,j in ti.ndrange(grid_n,grid_n):
        sum = 0
        for k in range(grid_n):
            sum += grain_count[i,j,k]
        column_sum[i,j] = sum

    accum_sum = 0
    ti.loop_config(serialize=True)
    for i in range(grid_n):
        for j in range(grid_n):
            prefix_sum[i,j,0] = accum_sum
            accum_sum += column_sum[i,j]

    for i,j in ti.ndrange(grid_n,grid_n):
        for k in range(grid_n):
            if k == 0:
                prefix_sum[i,j,k] += grain_count[i,j,k]
            else:
                prefix_sum[i,j,k] = prefix_sum[i,j,k-1] + grain_count[i,j,k]
            # assign list head, cur, tail
            # grid ID: i*grid_n*grid_n + j*grid_n + k
            linear_idx = i*grid_n*grid_n + j*grid_n + k
            list_head[linear_idx] = prefix_sum[i,j,k] - grain_count[i,j,k]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i,j,k]

    for i in range(total_number):
        grid_idx = ti.floor((tbf[i].p + 0.3) / grid_size, int)
        linear_idx = grid_idx[0]*grid_n*grid_n + grid_idx[1]*grid_n + grid_idx[2]
        grain_location = ti.atomic_add(list_cur[linear_idx],1)
        particle_id[grain_location] = i

    for i in range(total_number):
        grid_idx = ti.floor((tbf[i].p + 0.3) / grid_size, int)
        x_begin = max(grid_idx[0] - 1,0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1,0)
        y_end = min(grid_idx[1] + 2, grid_n)

        z_begin = max(grid_idx[2] - 1,0)
        z_end = min(grid_idx[2] + 2, grid_n)

        for neigh_i,neigh_j,neigh_k in ti.ndrange((x_begin,x_end),(y_begin,y_end),(z_begin,z_end)):
            linear_idx = neigh_i*grid_n*grid_n + neigh_j*grid_n + neigh_k
            for p_location in range(list_head[linear_idx],list_tail[linear_idx]):
                j = particle_id[p_location]
                if i<j:
                    resolve(i,j)

    for i in range(number):
        bf[i] = tbf[i]

@ti.kernel
def rotate():
    # spin matrix
    romat = ti.Matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], ti.float32)
    _spin = 0.001
    romat[0,0] = ti.cos(_spin)
    romat[2,0] = -ti.sin(_spin)
    romat[0,2] = ti.sin(_spin)
    romat[2,2] = ti.cos(_spin)
    romat[1,1] = 1
    for i in range(fix_n):
        fix_bf[i].p[2] += 0.2
        fix_bf[i].p = romat @ fix_bf[i].p
        fix_bf[i].p[2] -= 0.2
        tot_bf[number+i] = fix_bf[i]

# initial window, canvas, scene, camera
window = ti.ui.Window("3D GeoDEM",(640,640),show_window=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1.0,0,0.4)
camera.lookat(0,0,0)
camera.up(0,0,1)

# set ball color
ball_color = ti.Vector.field(3,ti.float32,number)
@ti.kernel
def assign_color():
    for i in range(number):
        if bf[i].p[0] > 0:
            ball_color[i] = [255/255,212/255,212/255]
        else:
            ball_color[i] = [255/255,110/255,0/255]
assign_color()
fix_ball_color = (10/255,212/255,212/255)

# define floor
floor = ti.Vector.field(3,ti.f32,6)
floor[0] = [-0.3,-0.3,-0.3]
floor[1] = [0.3,-0.3,-0.3]
floor[2] = [-0.3,0.3,-0.3]
floor[3] = [0.3,-0.3,-0.3]
floor[4] = [0.3,0.3,-0.3]
floor[5] = [-0.3,0.3,-0.3]
floor_color = (149/255,172/255,191/255)

while window.running:
    for _ in range(substeps):
        apply_bc()
        rotate()
        contact(tot_bf)
        update()
    # For rendering
    camera.track_user_inputs(window, movement_speed=0.001, yaw_speed=0.001, pitch_speed=0.001, hold_key=ti.ui.LMB)
    camera.position(1.0*math.cos(gts[None]*0.0001),1.0*math.sin(gts[None]*0.0001),0.4)
    scene.set_camera(camera)
    scene.ambient_light((0.3, 0.3, 0.3))
    scene.point_light(pos=(0,0,0.5), color=(1, 1, 1))
    scene.particles(centers=bf.p, per_vertex_color=ball_color , radius=0.00375)
    scene.particles(centers=fix_bf.p, color=fix_ball_color, radius=0.0009)
    scene.mesh(floor,color=floor_color)
    canvas.scene(scene)
    window.show()