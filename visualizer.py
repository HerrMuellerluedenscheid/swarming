from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as num

'''requires matplotlib > 1.4 due to 3D quiver'''

class Visualizer():
    def __init__(self, swarm):
        sources = swarm.get_sources()
        #x = swarm.geometry.xyz[0]
        #y = swarm.geometry.xyz[1],
        #z = swarm.geometry.xyz[2],
        times = []
        mags = []
        lats = []
        lons = []
        depths = []
        U = num.zeros(len(sources))
        V = num.zeros(len(sources))
        W = num.zeros(len(sources))
        for i, s in enumerate(sources):
            lats.append(s.effective_lat)
            lons.append(s.effective_lon)
            depths.append(s.depth)
            times.append(s.time)
            mags.append(5+s.magnitude*30)
            mt = s.pyrocko_moment_tensor()
            uvw = num.array(mt.t_axis())
            U[i] = uvw[0][0]
            V[i] = uvw[0][1]
            W[i] = uvw[0][2]
        fig = plt.figure() 
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(lons,
                        lats,
                        depths,
                        c=times,
                        s=mags,
                        marker='o')

        qv = ax.quiver(lons, lats, depths, U, V, W, length=0.01)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        #ax.set_xlim([-3000,3000])
        #ax.set_ylim([-3000,3000])
        #ax.set_zlim([-3000,3000])
        fig.colorbar(sc)

        plt.show()


