import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import random

G = 6.67e-11
sun_m = 1.989e30
earth_m = 5.972e24
earth_sun_d = 149.6e9
earth_iv = 30555.0
moon_d = 3.844e8
moon_v = 1023.0
moon_m = 7.348e22

crashed = False


class Sim:
    def __init__(self, step, time, speed_multiplier, bodies=[]):
        # Forces should be array of lambda
        self.step = step
        self.time = time
        self.speed_multiplier = speed_multiplier
        self.bodies = bodies

    def init_bodies(bodies):
        self.bodies = bodies


    def update_body_list(self):
        for idx, b in enumerate(self.bodies):
            b.update_pos(self.step, np.delete(self.bodies, idx))
        return self.bodies
    
    def calc_COM(self):
        totalMass = np.sum([a.mass for a in self.bodies])
        moms = [a.mass*a.pos for a in self.bodies]
        totalMom = [0.0, 0.0, 0.0]
        for m in moms:
            totalMom += m
        return totalMom/totalMass

    def update(self, i):

        for _ in range(self.speed_multiplier):
            self.update_body_list()


        content = ([a.pos[0] for a in self.bodies],
                   [a.pos[1] for a in self.bodies],
                   [a.pos[2] for a in self.bodies])

        self.graph._offsets3d = content
        comx, comy, comz = self.calc_COM()
        self.com_graph._offsets3d = ([comx],[comy],[comz])
        self.title.set_text('N-Body Sim, time={}'.format(i))

        self.ax.view_init(elev=20., azim=0.5*i)

    def animate(self):
        bodies = self.bodies
        step = self.step
        time = self.time
        speed_multiplier = self.speed_multiplier

        xlim=(-earth_sun_d, earth_sun_d)
        ylim=xlim
        zlim=xlim

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d', 
                xlim=xlim,
                ylim=ylim,
                zlim=zlim)
        self.graph = self.ax.scatter([a.pos[0] for a in bodies],
                           [a.pos[1] for a in bodies],
                           [a.pos[2] for a in bodies], s=100)
        com = self.calc_COM()
        self.com_graph = self.ax.scatter(com[0], com[1], com[2])
        self.title = self.ax.set_title('N-Body Sim')

        self.ani = animation.FuncAnimation(self.fig, self.update, frames=int(time/step),
                                  interval=20, blit=False)
        return self.ani

    def save_gif(self, name, artist="Alex Goodenbour"):
        Writer = animation.writers['imagemagick']
        writer = Writer(fps=15, metadata=dict(artist=artist), bitrate=1800)
        self.ani.save(name+'.gif', writer=writer)

    def show(self):
        return plt.show()

def grav_force(m1, m2, r1, r2):
    dist = r2-r1
    distMag = np.sqrt(dist.dot(dist))
    return (G*m1*m2/distMag**3)*dist




class Body:
    def __init__(self, mass, pos, vel, forces=[grav_force]):
        x, y, z = pos
        vx, vy, vz = vel
        self.mass = mass
        self.pos = np.array([x,y,z])
        self.a = np.array([0,0,0])
        self.v = np.array([vx,vy,vz])
        self.F = np.array([0,0,0])
        self.forces = forces

    def update_F(self, others):
        global crashed
        F = np.array([0,0,0])
        for other in others:
            for force in self.forces:
                F = F + force(self.mass, other.mass, self.pos, other.pos) 

            dist = other.pos-self.pos
            distMag = np.sqrt(dist.dot(dist))
            if distMag < 1e10:
                F = 0.0
                if not crashed:
                    print("Crash!")
                    crashed = True
        self.F = F
        return self.F

    def update_a(self, others):
        self.update_F(others)
        self.a = self.F/self.mass
        return self.a

    def update_v(self, step, others):
        self.update_a(others)
        self.v = self.v+step*self.a
        return self.v

    def update_pos(self, step, others):
        self.update_v(step, others)
        self.pos = self.pos+step*self.v
        return self.pos



earth_sun_sys = np.array([

    Body(sun_m,
        (0.0, 0.0, 0.0),
        (0.0, earth_iv*0.05, earth_iv*0.1)
        ), 
    Body(earth_m,
        (earth_sun_d, 0.0, 0.0), 
        (0.0, -earth_iv, 0.0)
        )
    # Body(earth_m,
        # ((11/10)*earth_sun_d, 0.0, 0.0), 
        # (0.0, (11/10)*earth_iv, 0.0)
        # )
    ])

sys2 = np.array([

    Body(sun_m,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.2*earth_iv)
        ), 
    Body(earth_m,
        (earth_sun_d, 0.0, 0.0), 
        (0.0, earth_iv, 0.0)
        ), 
    Body(earth_m,
        (0.0, earth_sun_d, 0.0), 
        (0.0, 0.0, 0.0)
        )
    ])

sys3 = np.array([Body(earth_m, 
                    (earth_sun_d*(random.random()-0.5),
                    earth_sun_d*(random.random()-0.5),
                    earth_sun_d*(random.random()-0.5)),
                    (0.0,0.0,0.0)) for _ in range(10)])
sys3 = np.append(sys3, Body(sun_m, 
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.3*earth_iv)))

sys4 = np.array([
    Body(sun_m,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0)
        ), 
    Body(sun_m,
        (0.0, 8*earth_sun_d, 2*earth_sun_d),
        (0.0, 0.0, 0.0)
        ), 
    Body(sun_m,
        (0.0, 0.0, 9*earth_sun_d),
        (0.0, 0.0, 0.0)
        ),
    Body(sun_m,
        (-4*earth_sun_d, -2*earth_sun_d, 0.0),
        (0.0, 0.0, 0.0)
        ) 
    ])










