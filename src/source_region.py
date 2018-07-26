import logging
from pyrocko.guts import Object, Float, Int, String, Bool
import numpy as num
from numpy import sin, cos
from pyrocko import moment_tensor, trace
from pyrocko.gf import seismosizer, Target, TriangularSTF
from scipy import stats, interpolate
from collections import defaultdict


logger = logging.getLogger(__name__)
uniform = num.random.uniform

sdr_ranges = dict(zip(['strike', 'dip', 'rake'], [[0., 360.],
                                                  [0., 90.],
                                                  [-180., 180.]]))

# Freely guessed values. keys-> magnitudes, values -> rise times
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


class BaseSourceGeometry(Object):
    center_lon = Float.T()
    center_lat = Float.T()
    center_depth = Float.T()
    azimuth = Float.T()
    dip = Float.T()
    tilt = Float.T()
    n = Int.T()

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


class Noise():
    def __init__(self, degap=False):
        self.degap = degap

    def make_noise(self, tr):
        return

    def apply(self, container):
        if not self.degap:
            for s, t, tr in container.iter():
                tr.add(self.noise[tr.nslc_id])

        else:
            traces = {}
            for s, t, tr in container.iter():
                if not t in traces.keys():
                    traces[t] = [tr]
                else:
                    traces[t].append(tr)

            noised = {}
            for t, trs in traces.items():
                tmin = min([tr.tmin for tr in trs])
                tmax = max([tr.tmax for tr in trs])
                degapped = trace.Trace(tmin=tmin,
                                       network=trs[0].network,
                                       station=trs[0].station,
                                       location=trs[0].location,
                                       channel=trs[0].channel,
                                       deltat=trs[0].deltat, 
                                       ydata=num.zeros((tmax-tmin)/trs[0].deltat))
                #degapped = trace.degapper(trs, maxgap=999999999999, fillmethod='zeros', deoverlap='add')
                for tr in trs:
                    degapped.add(tr)
                degapped.ydata += self.make_noise(degapped)
                noised[t] = degapped

            return noised


class GaussNoise(Noise):
    def __init__(self, degap=False, mu=1.):
        Noise.__init__(self, degap)
        self.mu = mu

    def make_noise(self, tr):
        return num.random.normal(0,self.mu,len(tr.ydata))


class CuboidSourceGeometry(BaseSourceGeometry):
    length = Float.T()
    depth = Float.T()
    thickness = Float.T()

    '''Source locations in a cuboid shaped volume'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info('setup geometry')
        z_range = num.array([-self.depth/2., self.depth/2.])
        x_range = num.array([-self.length/2., self.length/2.])
        y_range = num.array([-self.thickness/2., self.thickness/2.])

        _xyz = num.array([uniform(*x_range, size=self.n),
                          uniform(*y_range, size=self.n),
                          uniform(*z_range, size=self.n)])
        zs = uniform(*z_range, size=self.n)
        self.xyz = self.orientate(_xyz)


class GutenbergRichterDiscrete(Object):
    Mmin = Float.T(help='min magnitude')
    Mmax = Float.T(help='max magnitude')
    a = Float.T()
    b = Float.T()
    increment = Float.T(default=0.1, help='magnitude step')
    scaling = Float.T(default=1., help='factor scaling magnitudes')
    normalize = Bool.T(default=True)

    '''Magnitudes of events.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        x = num.arange(self.Mmin, self.Mmax+self.increment, self.increment)
        y = 10**(self.a - self.b*x)
        if self.normalize:
            y /= num.sum(y)

        x = x/self.increment

        data = (x, y)
        self.randomizer = stats.rv_discrete(name='magnitudes', values=data)

    def get_magnitude(self):
        return self.randomizer.rvs() * self.scaling

    def get_magnitudes(self, n):
        return self.randomizer.rvs(size=n) * self.scaling


class Timing(Object):
    tmin = Float.T()
    tmax = Float.T()

    ''' When should events fire?'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timings = {}

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
    def setup(self, swarm):
        logger.info('random timing')
        i = len(swarm.geometry.xyz.T)
        t = uniform(self.tmin, self.tmax, i)
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
    dip = Float.T(optional=True)
    azimuth = Float.T(optional=True)
    variance = Float.T(optional=True)

    def setup(self, swarm, dip=None, azimuth=None):
        logger.info('PropagationTiming')
        xyzs = swarm.geometry.xyz
        azimuth = swarm.geometry.azimuth if not self.azimuth else self.azimuth
        dip = swarm.geometry.dip if not self.dip  else self.dip
        b = unityvektorfromangles(azimuth, dip)
        npoints = len(xyzs.T)
        a_b = num.zeros(npoints)
        for i in range(npoints):
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


class FocalDistribution(Object):
    '''Randomizer class for focal mechanisms.
    Based on the base_source and a variation given in degrees focal mechanisms
    are randomized.'''

    n = Int.T(default=100)
    base_source = seismosizer.Source.T(optional=True)
    variation = Float.T(default=360)

    def __init__(self, **kwargs):
        '''
        :param n: number of needed focal mechansims
        :param base_source: a pyrocko reference source
        :param variation: [degree] by how much focal mechanisms may deviate
                        from *base_source*'''

        super().__init__(**kwargs)
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
                sdr = moment_tensor.random_strike_dip_rake(
                    s-self.variation/2., s+self.variation/2.,
                    d-self.variation/2., d+self.variation/2.,
                    r-self.variation/2., r+self.variation/2.)
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
        return cls(load_station_corrections(fn, revert=revert))


class Swarm(Object):
    '''This puts all privous classes into a nutshell and produces the swarm'''
    geometry = BaseSourceGeometry.T()
    timing = Timing.T()
    mechanisms = FocalDistribution.T()
    magnitudes = GutenbergRichterDiscrete.T()
    engine = seismosizer.Engine.T()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sources = []
        self.setup()

    def setup(self, model='dc'):
        '''Two different source types can be used: rectangular source and
        DCsource. When using DC Sources, the source time function (STF class)
        will be added using the *post_process* method. Otherwise, rupture
        velocity, rise time and rupture length are deduced from source depth
        and magnitude.'''
        # logger.info('Start setup swarm instance %s' % self)
        geometry = self.geometry.iter()
        center_depth = self.geometry.center_depth
        mechanisms = self.mechanisms.iter()
        self.timing.setup(self)
        timings = self.timing.iter()
        for north_shift, east_shift, depth in self.geometry.iter():
            logger.debug('%s, %s, %s'%(north_shift, east_shift, depth))
            mech = mechanisms.__next__()
            k, t = timings.__next__()
            mag = self.magnitudes.get_magnitude()
            z = depth + center_depth
            _vs = self.vs_from_depth(z)
            length, risetime = magnitude2risetimearea(mag, _vs)
            stf = TriangularSTF(duration=float(risetime))
            s = seismosizer.DCSource(lat=float(self.geometry.center_lat),
                                     lon=float(self.geometry.center_lon),
                                     depth=float(depth+center_depth),
                                     north_shift=float(north_shift),
                                     east_shift=float(east_shift),
                                     time=float(t),
                                     magnitude=float(mag),
                                     strike=float(mech[0]),
                                     dip=float(mech[1]),
                                     rake=float(mech[2]),
                                     stf=stf)
            s.validate()
            self.sources.append(s)

    def get_effective_sources(self):
        store = self.engine.get_store()
        logger.info('using store %s' % store.config.id)
        sources = [s for s in self.sources if \
            store.config.source_depth_min < s.depth < store.config.source_depth_max]
        nsources = len(self.sources)
        logger.info("removed %s of %s sources due to depth range." % (
            nsources - len(sources), nsources))
        return sources

    def get_events(self):
        '''Return a list of all events if depth within range of store.'''
        sources = self.get_effective_sources()
        events = []
        for i_s, s in enumerate(sources):
            s.name = str(i_s)
            events.append(s)
        return events

    def vs_from_depth(self, depth):
        """ linear interpolated vs at a given *depth* in the provided model."""
        store = self.engine.get_store()
        model = store.config.earthmodel_1d
        return model.material(depth).vs


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
    """Base class to define width, length and duration of source """
    def __init__(self, relation, model=None):
        self.model = model
        self.model_z = model.profile('z')
        self.model_vs = model.profile('vs')

    def vs_from_depth(self, depth):
        """ linear interpolated vs at a given *depth* in the provided model."""
        return self.model_vs.material(depth).vs

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

