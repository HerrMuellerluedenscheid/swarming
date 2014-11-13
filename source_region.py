import numpy as num
from numpy import sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyrocko import moment_tensor
from pyrocko.gf import seismosizer

def Rz(theta):
    return num.array([[1, 0 ,0 ],
                      [0, cos(theta), -sin(theta)],
                      [0, sin(theta), cos(theta)]])

def Ry(theta):
    return num.array([[cos(theta), 0 ,sin(theta) ],
                      [0, 1, -sin(theta)],
                      [-sin(theta), 0, cos(theta)]])

def Rx(theta):
    return num.array([[cos(theta), -sin(theta), 0. ],
                      [sin(theta), cos(theta), 0.],
                      [0, 0, 1]])

def rot_matrix(alpha, beta, gamma):
    '''rotations around z-, y- and x-axis'''
    alpha *= (num.pi/180.)
    beta *= (num.pi/180.)
    gamma *= (num.pi/180.)
    return num.dot(Rz(alpha), Ry(beta), Rx(gamma))


class BaseSourceGeometry():
    def __init__(self, center_lon, center_lat, center_depth, azimuth, dip, tilt, n):
        self.center_lon = center_lon
        self.center_lat = center_lat
        self.center_depth = center_depth
        self.azimuth = azimuth
        self.dip = dip
        self.tilt = tilt
        self.n = n
        self.xyz = None

    def orientate(self, xyz):
        '''Apply rotation and dipping.'''
        rm = rot_matrix(self.azimuth, self.dip, self.tilt)
        print rm
        _xyz = num.zeros(xyz.shape)
        _xyz = _xyz.T
        i = 0 
        for v in num.nditer(xyz, flags=['external_loop'], order='F'):
            _xyz[i] = num.dot(rm, v)
            i += 1
        return _xyz.T

    def setup(self):
        '''setup x, y and z coordinates of sources'''

    def iter_coordinates(self):
        for num.nditer(self.xyz, order='F'):
            yield F

        
class RectangularSourceGeometry(BaseSourceGeometry):
    def __init__(self, length, depth, thickness, *args, **kwargs):
        BaseSourceGeometry.__init__(self, *args, **kwargs)
        self.length = length
        self.depth = depth
        self.thickness = thickness
        self.setup()

    def setup(self):
        z_range = num.array([-self.depth/2., self.depth/2.])
        x_range = num.array([-self.length/2., self.length/2.])
        y_range = num.array([-self.thickness/2., self.thickness/2.])

        _xyz = num.array([num.random.uniform(*x_range, size=self.n),
                          num.random.uniform(*y_range, size=self.n),
                          num.random.uniform(*z_range, size=self.n)])
        zs = num.random.uniform(*z_range, size=self.n)
        self.xyz = self.orientate(_xyz)


class Timing:
    def __init__(self, tmin, tmax, events, distribution=None, geometry=None):
        self.geometry = geometry
        self.distribution = distribution
        self.tmin = tmin
        self.tmax = tmax
        self.n_steps = 10000

    def set_window(self):
        self.window = distribution

    def get_timings(self):
        if not self.geometry and not self.distribution:
            return num.random.normal(tmin, tmax, len(self.events))

class FocalDistribution():
    def __init__(self, n=0, base_mechanism=None, **kwargs):
        self.kwargs = kwargs
        self.n = n 
        self.base_mechanism = base_mechanism
        self.mechanisms = self.get_mechanisms(**kwargs)
    
    def get_mechanisms(self, **kwargs):
        '''
        kwargs: strikemin, strikemax, dipmin, dipmax, rakemin, rakemax 
        '''
        mechs = []
        if self.base_mechanism:
        
        for i in range(self.n_steps-1):
            mechs.append(moment_tensor.random_strike_dip_rake(**kwargs))
        return mechs

    def iter_mechanisms(self, **kwargs):
        for mech in list(self.get_mechanisms(**kwargs)):
            yield mech

class Swarm():
    def __init__(self, geometry, timing, focal_distribution):
        self.geometry = geometry
        self.timing = timing
        self.focal_distribution = focal_distribution
        self.source_list = seismosizer.SourceList()
    
    def setup(self):
        mechanisms = 
        for i in self.geometry.iter_coordinates():
            mech 
            self.source_list.append(s)






if __name__=='__main__':
    swarm = RectangularSourceRegion(center_lon=10, 
                                     center_lat=10.,
                                     center_depth=8000,
                                     azimuth=40.,
                                     dip=40.,
                                     tilt=45.,
                                     length=6000.,
                                     depth=6000.,
                                     thickness=500., 
                                     n=100)

    
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(swarm.xyz[0], 
               swarm.xyz[1],
               swarm.xyz[2],
               c='r',
               marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-3000,3000])
    ax.set_ylim([-3000,3000])
    ax.set_zlim([-3000,3000])

    plt.show()
