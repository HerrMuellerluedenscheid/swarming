from pyrocko.gf.seismosizer import RemoteEngine, Target
import numpy as num

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

# One way of requesting the processed traces is as follows
synthetic_traces = response.pyrocko_traces()


