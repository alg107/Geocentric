import Geocentric

step = 60*60
time = 60*60*24*5
speed_multiplier = 160

sim = Geocentric.Sim(step, time, speed_multiplier, Geocentric.earth_sun_sys, False)

sim.animate()
#sim.save_gif("falling")

sim.show()
