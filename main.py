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

class Body:
    def __init__(self, mass, pos, vel):
        x, y, z = pos
        vx, vy, vz = vel
        self.mass = mass
        self.pos = np.array([x,y,z])
        self.a = np.array([0,0,0])
        self.v = np.array([vx,vy,vz])
        self.F = np.array([0,0,0])

    def update_F(self, others):
        global crashed
        F = np.array([0,0,0])
        for other in others:
            dist = other.pos-self.pos
            distMag = np.sqrt(dist.dot(dist))
            F = F + (G*self.mass*other.mass/distMag**3)*dist
            if distMag < 10000000000:
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

def update_body_list(blist):
    for idx, b in enumerate(blist):
        b.update_pos(step, np.delete(blist, idx))
    return blist

earth_sun_sys = np.array([

    Body(sun_m,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0)
        ), 
    Body(earth_m,
        (earth_sun_d, 0.0, 0.0), 
        (0.0, earth_iv, 0.0)
        ), 
    Body(earth_m,
        ((11/10)*earth_sun_d, 0.0, 0.0), 
        (0.0, (11/10)*earth_iv, 0.0)
        )
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
        (9*earth_sun_d, 0.0, 0.0),
        (0.0, 0.0, 0.0)
        ) 
    ])


bodies = sys4
xlim=(-10*earth_sun_d, 10*earth_sun_d)
ylim=xlim
zlim=xlim

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',
        xlim=xlim,
        ylim=ylim,
        zlim=zlim)
graph = ax.scatter([a.pos[0] for a in bodies], [a.pos[1] for a in bodies], [a.pos[2] for a in bodies])
title = ax.set_title('N-Body Sim')
step = 60*60
time = 60*60*24*30*12

speed_multiplier = 200
def update(i):

    global bodies 

    for _ in range(speed_multiplier):
        update_body_list(bodies)


    content = ([a.pos[0] for a in bodies],
               [a.pos[1] for a in bodies],
               [a.pos[2] for a in bodies])

    graph._offsets3d = content
    title.set_text('N-Body Sim, time={}'.format(i))



if __name__=="__main__":
    # b1pos = []
    # b2pos = []
    # b3pos = []
    # for i in range(int(time/step)):
        # b1pos.append(bodies[0].pos)
        # b2pos.append(bodies[1].pos)
        # b3pos.append(bodies[2].pos)
        # update_body_list(bodies)
    # ax.plot([a[0] for a in b1pos], [a[1] for a in b1pos], color="blue")
    # ax.plot([a[0] for a in b2pos], [a[1] for a in b2pos], color="red")
    # ax.plot([a[0] for a in b3pos], [a[1] for a in b3pos], color="green")
    # plt.show()
    
    ani = animation.FuncAnimation(fig, update, frames=int(time/step),
                              interval=20, blit=False)
    plt.show()
