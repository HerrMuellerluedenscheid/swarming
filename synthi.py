from pyrocko.gf.seismosizer import RemoteEngine, Target, LocalEngine
from pyrocko.model import load_stations, dump_events
from pyrocko import io
from visualizer import Visualizer
from source_region import *
import numpy as num
import os

def guess_targets_from_stations(stations, channels='NEZ'):
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
                              codes=(s.network,
                                     s.station,
                                     s.location, 
                                     c)) for c in channels])
    return targets    
        
if __name__=='__main__':
    
    # I use the following environment variables to locate green function 
    # stores and the station files, which in this case are the webnet stations.
    webnet = os.environ["WEBNET"]
    stores = os.environ["STORES"]

    # Number of sources....
    number_sources = 20

    # swarm geometry
    geometry = RectangularSourceGeometry(center_lon=12.4, 
                                     center_lat=50.2,
                                     center_depth=10000,
                                     azimuth=40.,
                                     dip=10.,
                                     tilt=45.,
                                     length=2000.,
                                     depth=1000.,
                                     thickness=100., 
                                     n=number_sources)
    
    # reference source
    base_source = seismosizer.DCSource(lat=0, lon=0, depth=0, 
                                       strike=170, dip=80,rake=30)
    
    # Timing
    timing = RandomTiming(tmin=1000, tmax=100000, number=number_sources)
    #timing = PropagationTiming(geometry)
    
    # Focal Mechanisms based on reference source a variation of strike, dip 
    # and rake in degrees and the number of sources.
    mechanisms = FocalDistribution(n=number_sources, 
                                   base_source=base_source, 
                                   variation=20)

    # magnitude distribution
    magnitudes = MagnitudeDistribution.GutenbergRichter(a=1, b=1.0)

    # The store we are going extract green function from:
    store_id = 'vogtland'
    engine = LocalEngine(store_superdirs=[stores], 
                         default_store_id=store_id)

    # Obacht: das muss besser!
    store = engine.get_store()
    config = store.config 
    model = config.earthmodel_1d
    stf = STF(magnitude2risetimearea, model=model)

    # Gather these information to create the swarm:
    swarm = Swarm(geometry=geometry, 
                  timing=timing, 
                  mechanisms=mechanisms,
                  magnitudes=magnitudes,
                  stf=stf)

    # setup stations/targets:
    stats = load_stations(webnet+'/meta/stations.pf')
    
    # Scrutinize the swarm using matplotlib
    #Visualizer(swarm, stats)

    # convert loaded stations to targets (see function at the top).
    targets = guess_targets_from_stations(stats)

    # Processing that data will return a pyrocko.gf.seismosizer.Reponse object.
    response = engine.process(sources=swarm.get_sources(), 
                              targets=targets)

    
    # Save the events
    dump_events(swarm.get_events(), 'events_swarm.pf')
    io.save(response.pyrocko_traces(), 'swarm.mseed')

    convolved_traces = stf.post_process(response)
    
    # Save traces:
    io.save(convolved_traces.traces_list(), 'swarm_stf.mseed')
    # One way of requesting the processed traces is as follows
    #self.synthetic_traces = response.pyrocko_traces()
