import logging
import numpy as num
from numpy import sin, cos
import matplotlib.pyplot as plt
from pyrocko import moment_tensor
from pyrocko.gf import seismosizer, Target
from scipy import stats, interpolate
from collections import defaultdict
import os

logger = logging.getLogger(__name__)

sdr_ranges = dict(zip(['strike', 'dip', 'rake'], [[0., 360.],
                                                  [0., 90.],
                                                  [-180., 180.]]))
#Freely guessed values. keys-> magnitudes, values -> rise times
guessed_rise_times = {-1:0.001,
                      0:0.01,
                      1:0.05,
                      2:0.1,
                      3:0.2,
                      4:0.4,
                      5:0.6, 
                      6:1.0}

to_rad = num.pi/180.

def load_station_corrections(fn, combine_channels=True, revert=False):
    """
    :param combine_channels: if True, return one correction per station, which i
                             mean of all phases"""
    corrections = {}
    with open(fn, 'r') as f:
        for l in f.readlines():
            nslc_id, phasename, residual = l.split()
            nslc_id = tuple(nslc_id.split('.'))
            if not nslc_id in corrections.keys():
                corrections[nslc_id] = {}
            if residual=='None':
                residual = None
            else:
                residual = float(residual)
                if revert:
                    residual *= -1.

            corrections[nslc_id][phasename] = residual

    if combine_channels:
        combined = {}
        for nslc_id, phasename_residual in corrections.items():
            d = num.array(phasename_residual.values())
            d = d[d!=num.array(None)]
            combined[nslc_id[:3]] = num.mean(d)
        return combined
    else:
        return corrections


def guess_targets_from_stations(stations, channels='NEZ', quantity='velocity'):
    '''convert a list of pyrocko stations to individual seismosizer target 
    instances.'''
    targets = []
    for s in stations:
        if not s.channels:
            channels = channels
        else:
            channels = s.get_channels.keys()
       
        targets.extend([Target(lat=s.lat, 
                               lon=s.lon, 
                               elevation=s.elevation, 
                               depth=s.depth, 
                               quantity=quantity,
                              codes=(s.network,
                                     s.station,
                                     s.location, 
                                     c)) for c in channels])
    return targets    
    
    
def GutenbergRichterDiscrete(a,b, Mmin=0., Mmax=8., inc=0.1, normalize=True):
    """discrete GutenbergRichter randomizer.
    Use returnvalue.rvs() to draw random number. 
    
    :param a: a-value
    :param b: b-value
    :param Mmin: minimum magnitude of distribution
    :param Mmax: maxiumum magnitude of distribution
    :param inc: step of magnitudes """
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

def unityvektorfromangles(azi, dip):
    ax = num.cos(dip)*num.cos(azi)
    ay = num.cos(dip)*num.sin(azi)
    az = num.sin(dip)
    return num.array([ax, ay, az])

def proj(x, A): 
    '''x: vector to be projected onto A'''
    return num.dot(x, A)/num.dot(A,A)* A

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
        logger.info('apply geometry')
        #rm = rot_matrix(self.azimuth, self.dip, self.tilt)

        rm = moment_tensor.euler_to_matrix((90.-self.dip)/180.*num.pi,
                                           self.tilt/180.*num.pi, 
                                           self.azimuth/180.*num.pi)
        _xyz = num.zeros(xyz.shape)
        _xyz = _xyz.T
        i = 0 
        for v in num.nditer(xyz, flags=['external_loop'], order='F'):
            _xyz[i] = num.dot(rm, v)
            i += 1
        return _xyz.T

    def setup(self):
        '''setup x, y and z coordinates of sources'''
        pass

    def iter(self):
        for val in  num.nditer(self.xyz, order='F', flags=['external_loop']):
            yield val

class CuboidSourceGeometry(BaseSourceGeometry):
    '''Source locations in a cuboid shaped volume'''
    def __init__(self, length, depth, thickness, *args, **kwargs):
        BaseSourceGeometry.__init__(self, *args, **kwargs)
        self.length = length
        self.depth = depth
        self.thickness = thickness
        self.setup()

    def setup(self):
        logger.info('setup geometry')
        z_range = num.array([-self.depth/2., self.depth/2.])
        x_range = num.array([-self.length/2., self.length/2.])
        y_range = num.array([-self.thickness/2., self.thickness/2.])

        _xyz = num.array([num.random.uniform(*x_range, size=self.n),
                          num.random.uniform(*y_range, size=self.n),
                          num.random.uniform(*z_range, size=self.n)])
        zs = num.random.uniform(*z_range, size=self.n)
        self.xyz = self.orientate(_xyz)


class MagnitudeDistribution:
    '''Magnitudes that the events should have.'''
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
    ''' When should events fire?'''
    def __init__(self, tmin, tmax, *args, **kwargs):
        self.tmin = tmin
        self.tmax = tmax
        self.timings = {}
        self.count = -1

    def iter(self, key=None):
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
    ''' Random nucleation of events in a given time span

    :param tmin: start time
    :param tmax: stop time
    '''
    def __init__(self, *args, **kwargs):
        Timing.__init__(self, *args, **kwargs)
    
    def setup(self, swarm):
        logger.info('random timing')
        i = len(swarm.geometry.xyz.T)
        t = num.random.uniform(self.tmin, self.tmax, i)
        self.timings = dict(zip(range(i), t))
        
class PropagationTiming(Timing):
    '''
    Add a migration like behaviour to your swarm.

    :param dip: dipping angle of migration direction (optional)
    :param azimuth: azimuthal angle of migration direction (optional)
    :param variance: function to add variance in nucleation (optional)
    
    dip and azimuth are deduced from the orientation of the geometry if not
    explicitly given.
        
    :example:
        one_day = 3600.*24
        timing = PropagationTiming(
            tmin=0,
            tmax=one_day,
            dip=90,
            variance=lambda x:
                x+num.random.uniform(low=-one_day/2., high=one_day/2.))

    '''
    def __init__(self, *args, **kwargs):
        try:
            self.dip = kwargs.pop('dip')
        except KeyError:
            self.dip = None

        try:
            self.azimuth = kwargs.pop('azimuth')
        except KeyError:
            self.azimuth = None
        
        try:
            self.variance = kwargs.pop('variance')
        except KeyError:
            self.variance = None
        Timing.__init__(self, *args, **kwargs)
    
    def setup(self, swarm, dip=None, azimuth=None):
        logger.info('PropagationTiming')
        xyzs = swarm.geometry.xyz
        azimuth = swarm.geometry.azimuth if not self.azimuth else self.azimuth
        dip = swarm.geometry.dip if not self.dip  else self.dip
        b = unityvektorfromangles(azimuth, dip)
        npoints = len(xyzs.T)
        a_b = num.zeros(npoints)
        for i in xrange(npoints):
            a = xyzs.T[i]
            a_b[i] = num.linalg.norm(proj(a, b))
            if num.arccos(num.dot(a/num.linalg.norm(a),b/num.linalg.norm(b)))<num.pi/2.:
                a_b[i]*=-1

        min_ab = a_b.min()
        max_ab = a_b.max()

        t = min_ab+(self.tmax-self.tmin)/max_ab*a_b
        if self.variance:
            t = map(self.variance, t)
        
        self.timings = dict(zip(range(npoints), t))
        
    def __iter__(self):
        return iter(self.timings)

class FocalDistribution():
    '''Randomizer class for focal mechanisms. 
    Based on the base_source and a variation given in degrees focal mechanisms
    are randomized.'''
    def __init__(self, n=100, base_source=None, variation=360):
        '''
        :param n: number of needed focal mechansims
        :param base_source: a pyrocko reference source 
        :param variation: [degree] by how much focal mechanisms may deviate
                        from *base_source*'''
        self.n = n 
        self.base_source= base_source 
        self.variation = variation
        self.mechanisms = []
        self.setup_sources()

    def iter_mechanisms(self, **kwargs):
        for mech in list(self.get_mechanisms(**kwargs)):
            yield mech

    def setup_sources(self):
        logger.info('setup sources')
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

class Perturbation():
    def __init__(self, perturbations):
        self.perturbations = perturbations

    def apply(self, container):
        _got_perturbed = set()
        _not_got_perturbed = set()
        for s, t, tr in container.iter():
            trid = tr.nslc_id[:3]
            if trid in self.perturbations:
                tshift = self.perturbations[trid]
                tr.shift(tshift)
                _got_perturbed.add((trid, tshift))
            else:
                _not_got_perturbed.add(trid)
        logger.info('Perturbed:')
        [logger.info(trid) for trid in _got_perturbed]
        logger.info('NOT perturbed:')
        [logger.info(trid) for trid in _not_got_perturbed]

    @classmethod
    def from_file(cls, fn, revert=False):
        perturbations = load_station_corrections(fn, revert=revert)
        return cls(perturbations)


class Swarm():
    '''This puts all privous classes into a nutshell and produces the 
    swarm'''
    def __init__(self, geometry, timing, mechanisms, magnitudes, stf=None):
        self.geometry = geometry
        self.timing = timing
        self.mechanisms = mechanisms 
        self.magnitudes = magnitudes 
        self.sources = seismosizer.SourceList()
        self.stf = stf
        self.setup()

    def setup(self, model='dc'):
        '''Two different source types can be used: rectangular source and
        DCsource. When using DC Sources, the source time function (STF class)
        will be added using the *post_process* method. Otherwise, rupture
        velocity, rise time and rupture length are deduced from source depth
        and magnitude.'''
        logger.info('Start setup swarm instance %s'%self)
        geometry = self.geometry.iter()
        center_lat = self.geometry.center_lat
        center_lon = self.geometry.center_lon
        center_depth = self.geometry.center_depth

        mechanisms = self.mechanisms.iter()
        self.timing.setup(self)
        timings = self.timing.iter()
        if model=='rectangular':
            logger.debug('Nshift, Eshift, Z')
            for north_shift, east_shift, depth in self.geometry.iter():
                logger.debug('%s, %s, %s'%(north_shift, east_shift, depth))
                mech = mechanisms.next()
                k, t = timings.next()
                mag = self.magnitudes.get_magnitude()
                L, rt = self.stf.get_L_risetime(depth+center_depth, mag)
                z = depth+center_depth
                velo = self.stf.vs_from_depth(z+center_depth)
                s = seismosizer.CuboidSourceGeometry(lat=float(center_lat), 
                                                  lon=float(center_lon), 
                                                  depth=float(z),
                                                  north_shift=float(north_shift),
                                                  east_shift=float(east_shift),
                                                  time=float(t),
                                                  magnitude=float(mag),
                                                  strike=float(mech[0]),
                                                  dip=float(mech[1]),
                                                  rake=float(mech[2]),
                                                  length=float(L),
                                                  width=float(L),
                                                  velocity=float(velo),
                                                  risetime=float(t))
                s.validate()
        elif model=='dc':
            for north_shift, east_shift, depth in self.geometry.iter():
                logger.debug('%s, %s, %s'%(north_shift, east_shift, depth))
                #mech = mechanisms.next()
                mech = mechanisms.next()
                k, t = timings.next()
                mag = self.magnitudes.get_magnitude()
                #L, rt = self.stf.get_L_risetime(depth+center_depth, mag)
                z = depth+center_depth
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
        '''Return a list of source'''
        return self.sources

    def get_events(self):
        '''Return a list of all events'''
        return [s.pyrocko_event() for s in self.sources]


class Container():
    '''Similar to seismosizer.Response... Just for convenience'''
    def __init__(self):
        self.data = defaultdict(dict)
        self.targets = [] 
        self.sources = []

    def add_item(self, key1, key2, item):
        if key1 not in self.sources:
            self.sources.append(key1)
        if key2 not in self.targets:
            self.targets.append(key2)
        self.data[key1][key2] = item

    def traces_list(self):
        data = [] 
        for s, t_tr in self.data.items():
            for t, tr in t_tr.items():
                data.append(tr)
        return data

    def iter(self):
        for s, t_tr in self.data.items():
            for t, tr in t_tr.items():
                yield s, t, tr

    def __getitem__(self, key):
        return self.data[key]

    def snuffle(self):
        '''Open *snuffler* with requested traces.'''
        trace.snuffle(self.traces_list())


class STF():
    """Base class to define width, length and duartion of source """
    def __init__(self, relation, model=None):
        #self.relation = relation
        self.model = model
        self.model_z = model.profile('z')
        self.model_vs = model.profile('vs')

    def post_process(self, response, method='guess', chop_traces=True):
        '''a pyrocko.gf.seismosizer.Response object can be handed over on
        which source time functions are supposed to be applied.
        
        :param method: if this parameter is set to 'guess', 
            method guess_risettime_by_magnitude (see below) will be used
            to estimate rise times.
            Otherwise method magnitude2risetimearea is used. Latter one 
            is a scaling relation which is rather used for larger earthquakes.
        '''
        _return_traces = Container()
        _chop_speed_last = 2000.
        _chop_speed_first = 8000.
        for s,t,tr in response.iter_results():
            _vs = self.vs_from_depth(s.depth)
            if not method=='guess':
                length, risetime = magnitude2risetimearea(s.magnitude, _vs)
            else:
                risetime = guess_risettime_by_magnitude(s.magnitude)

            x_stf_new = num.arange(0.,risetime+tr.deltat, tr.deltat)
            if risetime<tr.deltat:
                if chop_traces:
                    dist = num.sqrt(s.distance_to(t)**2+s.depth**2)
                    tmax_last = dist/_chop_speed_last
                    tmax_first = dist/_chop_speed_first
                    tr.chop(tmin=s.time+tmax_first, tmax=s.time+tmax_last)
                    
                _return_traces.add_item(s, t, tr)
                continue
            
            finterp = interpolate.interp1d([0., x_stf_new[-1]*0.2, x_stf_new[-1]*0.8, x_stf_new[-1]], [0., 1., 1., 0.])
            y_stf_new = finterp(x_stf_new)
            #y_stf_new = num.zeros(len(x_stf_new)+20)
            y_stf_new[10:-10] = 1.
            if max(y_stf_new)!=1.:
                import pdb 
                pdb.set_trace()
            new_y = num.convolve(y_stf_new, tr.get_ydata(), 'same')
            tr.shift(x_stf_new[-1]*0.5)
            tr.set_ydata(new_y)
            if chop_traces:
                dist = num.sqrt(s.distance_to(t)**2+s.depth**2)
                tmax_last = dist/_chop_speed_last
                tmax_first = dist/_chop_speed_first
                tr.chop(tmin=s.time+tmax_first, tmax=s.time+tmax_last)
            
            if t.quantity=='velocity':
                a = tr.get_ydata()
                vel = num.append(a, 0)- num.append(0, a)
                tr.set_ydata(vel)
            
            _return_traces.add_item(s, t, tr)
    
        return _return_traces

    def vs_from_depth(self, depth):
        """ linear interpolated vs at a given *depth* in the provided model."""
        i_top_layer = num.max(num.where(self.model_z<=depth))
        i_bottom_layer = i_top_layer+1
        v_b = self.model_vs[i_bottom_layer]
        v_t = self.model_vs[i_top_layer]
        return v_t+(v_b-v_t)*(depth-self.model_z[i_top_layer])\
                /(self.model_z[i_top_layer]-self.model_z[i_bottom_layer])
        
    def get_L_risetime(self, depth, magnitude):
        _vs = self.vs_from_depth(depth)
        return magnitude2risetimearea(magnitude, _vs)


def guess_risettime_by_magnitude(mag):
    '''interpolate rise times from guessed rise times at certain magnitudes.
    modify the dictionary below at free will if you know which rise times are
    to be expected.'''
    
    mags = num.array(guessed_rise_times.keys())
    rts = num.array(guessed_rise_times.values())

    finterp = interpolate.interp1d(mags, rts)
    return finterp(mag)


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

