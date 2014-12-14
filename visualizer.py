from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as num

'''requires matplotlib > 1.4 due to 3D quiver'''

class Visualizer():
    def __init__(self, swarm, stations=None):
        sources = swarm.get_sources()
        #x = swarm.geometry.xyz[0]
        #y = swarm.geometry.xyz[1],
        #z = swarm.geometry.xyz[2],
        times = []
        mags = num.zeros(len(sources))
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
            mags[i] = s.magnitude
            mt = s.pyrocko_moment_tensor()
            uvw = num.array(mt.t_axis())
            U[i] = uvw[0][0]
            V[i] = uvw[0][1]
            W[i] = uvw[0][2]
        fig = plt.figure() 
        ax = fig.add_subplot(131, projection='3d')
        sc = ax.scatter(lons,
                        lats,
                        depths,
                        c=times,
                        s=(3.+mags)*30.,
                        marker='o')

        ax = fig.add_subplot(132, projection='3d')
        sc = ax.scatter(lons,
                        lats,
                        depths,
                        c=times,
                        s=(3.+mags)*30.,
                        marker='o')

        if stations:
            lats = []
            lons = []
            depths = []
            for s in stations:
                lats.append(s.lat)
                lons.append(s.lon)
                depths.append(s.depth)
            print len(depths), len(lats), len(lons)
            ax.scatter(num.array(lons),
                            num.array(lats),
                            num.array(depths),
                            c='b',
                            s=30,
                            marker='^')
                 
        #qv = ax.quiver(lons, lats, depths, U, V, W, length=0.01)
        ax.set_xlabel('Lon')
        ax.set_ylabel('Lat')
        ax.set_zlabel('Depth')
        ax.invert_zaxis()
        #ax.set_xlim([-3000,3000])
        #ax.set_ylim([-3000,3000])
        #ax.set_zlim([-3000,3000])
        fig.colorbar(sc)
        
        ax = fig.add_subplot(133)
        ax.hist(mags, bins=num.arange(min(mags), max(mags)+0.1, 0.1))
        plt.show()


