import numpy as num
import os

from pyrocko.gf.seismosizer import RemoteEngine, Target, LocalEngine, DCSource
from pyrocko.model import load_stations, dump_events
from pyrocko import io

from visualizer import Visualizer
from source_region import magnitude2risetimearea
from source_region import Swarm, CuboidSourceGeometry, RandomTiming, FocalDistribution, MagnitudeDistribution, STF
from noisify import Noise
from utils import get_targets, guess_targets_from_stations


if __name__=='__main__':

    # I use the following environment variables to locate green function 
    # stores and the station files, which in this case are the webnet stations.
    #webnet = os.environ["WEBNET"]
    webnet = '/data'
    #stores = os.environ["STORES"]
    stores = '/media/usb/stores/'

    # Number of sources....
    number_sources = 50

    # swarm geometry.
    # center_lat, center_lon and center_depth define the center point of
    # the desired swarm area.
    # The orientation is then given as follows.
    # azimuth is the angle versus north (rotation around z-axis), dip is the
    # the rotation around the horizontal y axis and tilt around x axis.
    # length, depth and thickness describe the x-, z- and y-extension
    geometry = CuboidSourceGeometry(center_lon=12.45, 
                                     center_lat=50.214,
                                     center_depth=10000,
                                     azimuth=-40.,
                                     dip=10.,
                                     tilt=25.,
                                     length=2000.,
                                     depth=1500.,
                                     thickness=100., 
                                     n=number_sources)

    # reference source. All other source mechanisms will be similar to this
    # one but with some degree of freedom defined by a deviation angle.
    # I guessed strike dip and rake angles based on the following paper
    # be Horalek and Sileny:
    # http://gji.oxfordjournals.org/content/194/2/979.full.pdf?keytype=ref&ijkey=bMwD67zn37OEiJs
    base_source = DCSource(lat=0, lon=0, depth=0, 
                                       strike=170, dip=80,rake=-30)

    # Timing tmin and tmax are in seconds after 1.1.1970
    one_day = 24*60*60
    timing = RandomTiming(tmin=0, tmax=2*one_day, number=number_sources)

    # The PropagationTiming class is not finished yet. The idea was to be
    # able to let the events start nucleating at one point and let them 
    # propagate through the medium. 
    #timing = PropagationTiming(geometry)

    # Focal Mechanisms based on reference source a variation of strike, dip 
    # and rake in degrees and the number of sources.
    mechanisms = FocalDistribution(n=number_sources, 
                                   base_source=base_source, 
                                   variation=25)

    # magnitude distribution with a- and b- value and a minimum magnitude.
    magnitudes = MagnitudeDistribution.GutenbergRichter(a=1, b=0.75, Mmin=0.)

    # The store we are going extract green functions from:
    #store_id = 'vogtland_50Hz_step'
    store_id = 'qplayground_total_1'
    engine = LocalEngine(store_superdirs=[stores], 
                         default_store_id=store_id)

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
    #stats = load_stations(webnet+'/meta/stations.pf')
    stats = load_stations('stations.pf')
    #stats = []
    # Scrutinize the swarm using matplotlib

    noise = Noise(files='/media/usb/webnet/mseed/2008')

    # convert loaded stations to targets (see function at the top).
    #targets = guess_targets_from_stations(stats)
    targets = get_targets(stations, noise.data_pile, store_id=store_id)
    Visualizer(swarm, stats)

    # Processing that data will return a pyrocko.gf.seismosizer.Reponse object.
    response = engine.process(sources=swarm.get_sources(),
                              targets=targets)

    # Save the events
    dump_events(swarm.get_events(), 'events_swarm.pf')
    io.save(response.pyrocko_traces(), 'swarm.mseed')

    convolved_traces = stf.post_process(response)

    # Save traces:
    io.save(convolved_traces.traces_list(), 'swarm_stf.mseed')
