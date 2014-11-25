from pyrocko.gf.seismosizer import RemoteEngine, Target, LocalEngine
from pyrocko.model import load_stations
from visualizer import Visualizer
from source_region import *
import numpy as num

def guess_targets_from_stations(stations, channels='NEZ'):
    targets = []
    for s in stations:
        if not s.channel:
            channels = channels
        else:
            channel = s.channel
        Target(s.lat, s.lon, s.elevation, s.depth, codes=channel)
    return targets    
        
if __name__=='__main__':

    number_sources = 100

    # swarm geometry
    geometry = RectangularSourceGeometry(center_lon=10, 
                                     center_lat=10.,
                                     center_depth=8000,
                                     azimuth=40.,
                                     dip=40.,
                                     tilt=45.,
                                     length=6000.,
                                     depth=6000.,
                                     thickness=500., 
                                     n=number_sources)
    
    # reference source
    base_source = seismosizer.DCSource(lat=0, lon=0, depth=0, 
                                       strike=170, dip=80,rake=30)
    
    # Timing
    timing = RandomTiming(tmin=1000, tmax=100000, number=number_sources)
    
    # Focal Mechanisms 
    mechanisms = FocalDistribution(n=number_sources, 
                                   base_source=base_source, 
                                   variation=20)

    #magnitude distribution
    magnitudes = MagnitudeDistribution.GutenbergRichter(a=1, b=1.0)

    swarm = Swarm(geometry=geometry, 
                 timing=timing, 
                 mechanisms=mechanisms,
                 magnitudes=magnitudes,
                 )

    engine = LocalEngine(store_superdirs=['/data/stores'])

    # The store we are going extract data from:
    store_id = 'crust2_dd'
    
    # setup stations/targets:
    stats = load_stations('/data/webnet/meta/stations.pf')
    targets = guess_targets_from_stations(stats)
    #targets = [Target(lat=10.1,
    #                  lon=10.1,
    #                  elevation=0.,
    #                  codes=('', 'SYN', '', channel_id)
    #                  )  for channel_id in 'NEZ']

    # Processing that data will return a pyrocko.gf.seismosizer.Reponse object.
    response = engine.process(sources=swarm.get_sources(), 
                              targets=targets)

    # One way of requesting the processed traces is as follows
    #self.synthetic_traces = response.pyrocko_traces()

    #Visualizer(swarm)



