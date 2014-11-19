from pyrocko.gf.seismosizer import RemoteEngine, Target, LocalEngine
from visualizer import Visualizer
from source_region import *
import numpy as num


if __name__=='__main__':

    number_sources = 100
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
    
    base_source = seismosizer.DCSource(lat=0, lon=0, depth=0, 
                                       strike=170, dip=80,rake=30)
    
    timing = RandomTiming(tmin=1000, tmax=100000, number=number_sources)
    
    mechanisms = FocalDistribution(n=number_sources, 
                                   base_source=base_source, 
                                   variation=20)

    magnitudes = MagnitudeDistribution.GutenbergRichter(a=1, b=1.0)
    
    swarm = Swarm(geometry=geometry, 
                 timing=timing, 
                 mechanisms=mechanisms,
                 magnitudes=magnitudes)

    engine = LocalEngine(store_superdirs=['/data/stores'])

    # The store we are going extract data from:
    store_id = 'crust2_dd'

    # Processing that data will return a pyrocko.gf.seismosizer.Reponse object.
    response = engine.process(sources=swarm.get_sources(), 
                              targets=targets)

    # One way of requesting the processed traces is as follows
    self.synthetic_traces = response.pyrocko_traces()

    #Visualizer(swarm)



