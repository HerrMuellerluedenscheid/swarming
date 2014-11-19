import numpy as num
from numpy import sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyrocko import moment_tensor
from pyrocko.gf import seismosizer

sdr_ranges = dict(zip(['strike', 'dip', 'rake'], [[0., 360.],
                                                  [0., 90.],
                                                  [-180., 180.]]))


def GutenbergRichterDiscrete(a,b, Mmin=0., Mmax=8., inc=0.1, normalize=True):
    """discrete GutenbergRichter randomizer.
    Use returnvalue.rvs() to draw random number"""
    x = num.arange(Mmin, Mmax+inc, inc)
    y = 10**(a-b*x)
    if normalize:
        y/=num.sum(y)
    
    return x/inc, y, inc 


def GR_distribution(a,b, mag_lo, mag_hi):
    '''from hazardlib or similar?! 
    TODO: wirklich gebraucht? '''
    return 10**(a- b*mag_lo) - 10**(a - b*mag_hi)

def GR(a, b, M):
    return 10**(a-b*M) 

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
        for val in  num.nditer(self.xyz, order='F'):
            yield val

        
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


class MagnitudeDistribution:
    def __init__(self, x, y, Mmin=0., Mmax=8., inc=0.1, scaling=1.):
        self.x = x
        self.y = y
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.inc = inc
        self.scaling = scaling
        self.update_randomizer()

    def set_distribution(self, x, y, scaling=1):
        self.scaling = scaling
        self.x = x
        self.y = y
        self.update_randomizer()

    def get_randomizer(self):
        return self.randomizer

    def update_randomizer(self):
        data = (self.x, self.y)
        self.randomizer = stats.rv_discrete(name='magnitues', values=data)

    def get_magnitude(self):
        return self.randomizer.rvs()*self.scaling

    def get_magnitudes(self, n):
        return self.randomizer.rvs(size=n)*self.scaling

    @classmethod
    def GutenbergRichter(cls, *args, **kwargs):
        x, y, scaling = GutenbergRichterDiscrete(*args, **kwargs)
        return cls(x,y, scaling=scaling)


class Timing:
    def __init__(self, tmin, tmax, *args, **kwargs):
        self.tmin = tmin
        self.tmax = tmax
        self.timings = {}

    def iter_timings(self):
        for k,t in self.timings.iter_items():
            yield k, t

    def get_timings(self):
        return self.timings 

    def setup(self):
        pass


class RandomTiming(Timing):
    def __init__(self, *args, **kwargs):
        Timing.__init__(self, *args, **kwargs)
        self.setup(kwargs.get['number'])

    def setup(self, number):
        i = num.range(number)
        t = num.random.uniform(self.tmin, self.tmax, number)
        self.timings = dict(zip(i, t))


class FocalDistribution():
    def __init__(self, n=100, base_mechanism=None, variation=360, magnitude_distribution=None):
        self.n = n 
        self.base_mechanism = base_mechanism
        self.variation = variation
        self.mag_dist = magnitude_distribution
        self.sources = []
        self.setup_sources()

    def iter_mechanisms(self, **kwargs):
        for mech in list(self.get_mechanisms(**kwargs)):
            yield mech

    def setup_sources(self):
        bs = self.base_source
        s,d,r = bs.strike, bs.dip, bs.rake
        for i in range(self.n):
            sdr = moment_tensor.random_strike_dip_rake(s-self.variation/2.,
                                                       s+self.variation/2.,
                                                       d-self.variation/2.,
                                                       d+self.variation/2.,
                                                       r-self.variation/2.,
                                                       r+self.variation/2.)

            self.sources.append(seismosizer.DCSource(lat=0., 
                                                     lon=0., 
                                                     depth=0., 
                                                     strike=sdr[0], 
                                                     dip=sdr[1], 
                                                     rake=sdr[2]))

    def iter_sources(self):
        for s in self.srcs:
            yield s

    def get_sources(self):
        return self.sources


class Swarm():
    def __init__(self, geometry, timing, mechanisms):
        self.geometry = geometry
        self.timing = timing
        self.focal_distribution = mechanisms 
        self.source_list = seismosizer.SourceList()
        self.setup_sources()


    def setup(self):
        mechanisms = self.focal_distribution.iter_mechanisms()
        timings = self.timing.iter_timings()
        for i in self.geometry.iter_coordinates():
            mech = mechanisms.next()
            t = timings.next()
            self.source_list.append(s)




if __name__=='__main__':
    geometry = RectangularSourceGeometry(center_lon=10, 
                                     center_lat=10.,
                                     center_depth=8000,
                                     azimuth=40.,
                                     dip=40.,
                                     tilt=45.,
                                     length=6000.,
                                     depth=6000.,
                                     thickness=500., 
                                     n=100)

    timing = 

    
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(geometry.xyz[0], 
               geometry.xyz[1],
               geometry.xyz[2],
               c='r',
               marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-3000,3000])
    ax.set_ylim([-3000,3000])
    ax.set_zlim([-3000,3000])

    plt.show()
