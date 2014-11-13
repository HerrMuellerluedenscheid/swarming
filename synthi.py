from pyrocko.gf.seismosizer import RemoteEngine, Target
from pyrocko.gf.seismosizer import DCSource, ExplosionSource
from pyrocko import trace
import numpy as num


def rot_matrix(theta, phi):
    theta = theta/180.*num.pi
    phi = phi/180.*num.pi
    t = theta
    p = phi
    return num.array([[sin(t)*cos(p), cos(t)*cos(p), -sin(p)],
                     [sin(t)*sin(p), cos(t)*sin(p), cos(p)],
                     [cos(t), -sin(t), 0.]])

class BaseSourceRegion():
    def __init__(self, center_lon, center_lat, center_depth, azimuth, dip):
        self.center_lon = center_lon
        self.center_lat = center_lat
        self.center_depth = center_depth
        self.azimuth = azimuth
        self.dip = dip

        self.setup()
    
    def orientate(self):
        '''Apply rotation and dipping.'''
        rm = rot_matrix(self.azimuth, self.dip)
        return num.dot(self.euler_coords, rm)

    def coordinate(self):
        self.orientate()

    def setup(self):
        '''setup x, y and z coordinates of sources'''

        
class RectrangularSourceRegion(BaseSourceRegion):
    def __init__(self, top_length, side_length, thickness, *args, **kwargs):
        BaseSourceRegion.__init__(self, *args, **kwargs)

    def setup(self):
        z_range = [self.center_depth-side_length/2., 
                  self.center_depth+side_length/2.]
        x_range = [-top_length/2., top_length/2.]
        y_range = [-thickness/2., thickness/2.]

        self.x = num.random.uniform(*x_range)
        self.z = num.random.uniform(*z_range)
        self.y = num.random.uniform(*y_range)


class swarm():
    def __init__(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax
        self.geometry = None

    def ellipse(self, a, b, s):
        return 

    def rectangle(



# We need a pyrocko.gf.seismosizer.Engine object which provides us with the 
# traces extracted from the store. In this case we are going to use a local 
# engine since we are going to query a local store.
engine = LocalEngine(store_superdirs=['/data/stores'])

# The store we are going extract data from:
store_id = 'crust2_dd'

# Define a list of pyrocko.gf.seismosizer.Target objects, representing the 
# recording devices. In this case one station with a three component sensor 
# will serve fine for demonstation. 
channel_codes = 'ENZ'
targets = [Target(lat=10., 
                  lon=10., 
                  store_id=store_id,
                  codes=('', 'STA', '', channel_code))
                        for channel_code in channel_codes]

# Let's use a double couple source representation.
source_dc = DCSource(lat=11.,
                     lon=11.,
                     depth=10000.,
                     strike=20.,
                     dip=40.,
                     rake=60.,
                     magnitude=4.)

# Processing that data will return a pyrocko.gf.seismosizer.Reponse object.
response = engine.process(sources=[source_dc], 
                          targets=targets)

print response
# One way of requesting the processed traces is as follows
synthetic_traces = response.pyrocko_traces()


