import numpy as num
from numpy import sin, cos
import matplotlib.pyplot as plt
from pyrocko import moment_tensor
from pyrocko.gf import seismosizer
from scipy import stats, interpolate
import os

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
        _xyz = num.zeros(xyz.shape)
        _xyz = _xyz.T
        i = 0 
        for v in num.nditer(xyz, flags=['external_loop'], order='F'):
            _xyz[i] = num.dot(rm, v)
            i += 1
        return _xyz.T

    def setup(self):
        '''setup x, y and z coordinates of sources'''

    def iter(self):
        for val in  num.nditer(self.xyz, order='F', flags=['external_loop']):
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
        self.randomizer = stats.rv_discrete(name='magnitudes', values=data)

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

    def iter(self, key=None):
        #if key:
        #    keys = sorted(self.timings.keys())
        #else:
        keys = sorted(self.timings.keys())
            
        for k in keys:
            yield k, self.timings[k]

    def set_key(self, key):
        self.key = key

    def get_timings(self):
        return self.timings 

    def setup(self):
        pass


class RandomTiming(Timing):
    def __init__(self, *args, **kwargs):
        Timing.__init__(self, *args, **kwargs)
        self.setup(kwargs.get('number'))

    def setup(self, number):
        i = range(number)
        t = num.random.uniform(self.tmin, self.tmax, number)
        self.timings = dict(zip(i, t))

class AzimuthalPropagation(Timing):
    def __init__(self, *args, **kwargs):
        Timing.__init__(self, *args, **kwargs)
        self.setup(kwargs.get('number'))

    def setup(number, sources, azimuth, dip, rake):
        # projection onto rupture vector
        perp = (azimuth+90.)/180.*num.pi

        # linear interpolation 

        # adding normal randomization



class FocalDistribution():
    def __init__(self, n=100, base_source=None, variation=360):
        self.n = n 
        self.base_source= base_source 
        self.variation = variation
        self.mechanisms = []
        self.setup_sources()

    def iter_mechanisms(self, **kwargs):
        for mech in list(self.get_mechanisms(**kwargs)):
            yield mech

    def setup_sources(self):
        if self.base_source:
            bs = self.base_source
            s,d,r = bs.strike, bs.dip, bs.rake
        for i in range(self.n):
            if self.base_source:
                sdr = moment_tensor.random_strike_dip_rake(s-self.variation/2.,
                                                       s+self.variation/2.,
                                                       d-self.variation/2.,
                                                       d+self.variation/2.,
                                                       r-self.variation/2.,
                                                       r+self.variation/2.)
            else:
                sdr = moment_tensor.random_strike_dip_rake()
            self.mechanisms.append(sdr)

    def iter(self):
        for mech in self.mechanisms:
            yield mech

    def get_mechanisms(self):
        return self.mechanisms


class Swarm():
    def __init__(self, geometry, timing, mechanisms, magnitudes):
        self.geometry = geometry
        self.timing = timing
        self.mechanisms = mechanisms 
        self.magnitudes = magnitudes 
        self.sources = seismosizer.SourceList()
        self.setup()

    def setup(self):
        geometry = self.geometry.iter()
        center_lat = self.geometry.center_lat
        center_lon = self.geometry.center_lon
        center_depth = self.geometry.center_depth

        mechanisms = self.mechanisms.iter()
        timings = self.timing.iter()
        for north_shift, east_shift, depth in self.geometry.iter():
            mech = mechanisms.next()
            k, t = timings.next()
            mag = self.magnitudes.get_magnitude()
            s = seismosizer.DCSource(lat=float(center_lat), 
                                     lon=float(center_lon), 
                                     depth=float(depth+center_depth),
                                     north_shift=float(north_shift),
                                     east_shift=float(east_shift),
                                     time=float(t),
                                     magnitude=float(mag),
                                     strike=float(mech[0]),
                                     dip=float(mech[1]),
                                     rake=float(mech[2]))
            s.validate()
            self.sources.append(s)

    def get_sources(self):
        return self.sources

class STF():
    """Base class to define width, length and duartion of source """
    def __init__(self, relation, model=None):
        self.relation = relation
        self.model = model
        self.model_z = model.profile('z')
        self.model_vs = model.profile('vs')

    def process(self, response):
        for s,t,tr in response.iter_results():
            _vs = self.vs_from_depth(s.depth)
            length, risetime = magnitude2risetimearea(s.magnitude, _vs)
            slip = num.arange(0., risetime, tr.deltat)
            finterp = interpolate.interp1d([0.,risetime], [0., 1.])
            ynew = finterp(slip)
            tr.set_ydata(num.convolve(ynew, tr.get_ydata()))
        response.snuffle()
    
        return response

    def vs_from_depth(self, depth):
        """ linear interpolated vs at *depth*"""
        i_top_layer = num.max(num.where(self.model_z<=depth))
        i_bottom_layer = i_top_layer+1
        v_b = self.model_vs[i_bottom_layer]
        v_t = self.model_vs[i_top_layer]
        return v_t+(v_b-v_t)*(depth-self.model_z[i_top_layer])\
                /(self.model_z[i_top_layer]-self.model_z[i_bottom_layer])
        

def magnitude2risetimearea(mag, vs):
    """Following Hank and Bakun 2002 and 2008
    http://www.opensha.org/glossary-magScalingRelation
    I assume rectangular source model. Rupture velocity 0.9*vs"""
    # area
    a = num.exp(mag-3.98)
    #side length [m] rectangular source plane
    length = num.sqrt(a)*1000
    risetime = length/(0.9*vs)
    return length, risetime

