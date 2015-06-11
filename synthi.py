import numpy as num
import os
import progressbar
import logging

from pyrocko.gf.seismosizer import RemoteEngine, Target, LocalEngine, Request, Response
from pyrocko.model import load_stations, dump_events
from pyrocko import io, trace
from visualizer import Visualizer
from source_region import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def dir_from_event(e):
    return e.time_as_string().replace(' ', '_').replace(':', '')

def do_pad_traces(trs, fade=0.1):
    t_min = min(trs, key=lambda t: t.tmin).tmin
    t_max = max(trs, key=lambda t: t.tmax).tmax
    fader = trace.CosFader(xfrac=fade)
    for tr in trs:
        tr.taper(fader)
        tr.extend(tmin=t_min, tmax=t_max)

    return trs

def write_container_to_dirs(container, base_dir, pad_traces=True):
    sources = container.sources
    targets = container.targets

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    p = progressbar.ProgressBar(widgets=['writing: ',
                                         progressbar.Percentage(), 
                                         progressbar.Bar()],
                                maxval=len(sources)).start()
    for i_s, s in enumerate(sources):
        _trs = [container[s][t] for t in targets]
        if pad_traces:
            _trs = do_pad_traces(_trs)
        e = s.pyrocko_event()
        e.set_name(str(i_s))
        out_dir = os.path.join(base_dir, dir_from_event(e))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for tr in _trs:
            io.save(tr, filename_template=out_dir+'/tr_%(network)s.%(station)s.%(location)s.%(channel)s.mseed')

        e.olddump(os.path.join(out_dir, 'event.pf'))

        p.update(i_s)
    p.finish()

if __name__=='__main__':
    
    # I use the following environment variables to locate green function 
    # stores and the station files, which in this case are the webnet stations.
    
    
    webnet = os.environ["WEBNET"]
    stores = os.environ["STORES"]

    # Number of sources....
    number_sources = 200

    # swarm geometry.
    # center_lat, center_lon and center_depth define the center point of
    # the desired swarm area.
    # The orientation is then given as follows.
    # azimuth is the angle versus north (rotation around z-axis), dip is the
    # the rotation around the horizontal y axis and tilt around x axis.
    # length, depth and thickness describe the x-, z- and y-extension
    geometry = CuboidSourceGeometry(center_lon=12.45, 
                                     center_lat=50.214,
    #geometry = CuboidSourceGeometry(center_lon=0., 
    #                                 center_lat=0.,
                                     center_depth=10000,
                                     azimuth=15,
                                     dip=-75.,
                                     tilt=0.,
                                     length=3000.,
                                     depth=3000.,
                                     thickness=200., 
                                     n=number_sources)
    
    # reference source. All other source mechanisms will be similar to this
    # one but with some degree of freedom defined by a deviation angle.
    # I guessed strike dip and rake angles based on the following paper
    # be Horalek and Sileny:
    # http://gji.oxfordjournals.org/content/194/2/979.full.pdf?keytype=ref&ijkey=bMwD67zn37OEiJs
    base_source = seismosizer.DCSource(lat=0, lon=0, depth=0, 
                                       strike=170, dip=80,rake=-30)
    
    # Timing tmin and tmax are in seconds after 1.1.1970
    #one_day = 24*60*60
    #timing = RandomTiming(tmin=0, tmax=2*one_day, number=number_sources)

    # The PropagationTiming class is not finished yet. The idea was to be
    # able to let the events start nucleating at one point and let them 
    # propagate through the medium. 
    one_day = 3600.*24
    timing = PropagationTiming(
        tmin=0,
        tmax=one_day,
        dip=45,
        variance=lambda x:
            x+num.random.uniform(low=-one_day/1.2, high=one_day/1.2))
    
    # Focal Mechanisms based on reference source a variation of strike, dip 
    # and rake in degrees and the number of sources.
    mechanisms = FocalDistribution(n=number_sources, 
                                   base_source=base_source, 
                                   variation=30)

    # magnitude distribution with a- and b- value and a minimum magnitude.
    magnitudes = MagnitudeDistribution.GutenbergRichter(a=1, b=0.5, Mmin=0.5)

    # The store we are going extract green functions from:
    #store_id = 'vogtland_50Hz_step'
    #store_id = 'vogtland_7'
    store_id = 'vogtland_fischer_horalek_2000_vpvs169_minus4p'
    #store_id = 'vogtland_malek2004_alexandrakis_100'
    engine = LocalEngine(use_config=True,
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
    stats = load_stations(webnet+'/meta/stations.pf')
    #stats = load_stations('/home/des/karamzad/working/myprograms/project/meta/stations_vogtland_all.txt')
    corrections = os.path.join(os.environ['HOME'],
                               'src/seismerize',
                               'residuals_median_CakeResiduals.dat')
    
    # Add a perturbation. *revert=True* means, that the traces are shifted by the correction time * -1.
    perturbation = Perturbation.from_file(corrections, revert=True)

    # convert loaded stations to targets (see function at the top).
    targets = guess_targets_from_stations(stats, quantity='velocity')
    logger.info('processing request...')
    print 'SKIP!'
    # Processing that data will return a pyrocko.gf.seismosizer.Reponse object.
    response = engine.process(sources=swarm.get_sources(), 
                              targets=targets)
    #logger.info('done')

    ## Save the events
    #events= swarm.get_events()
    #for i, e in enumerate(events):
    #    e.set_name(str(i))

    #dump_events(events, 'events_swarm.pf')
    #io.save(response.pyrocko_traces(), 'swarm.mseed')
    print 'convolve'
    convolved_traces = stf.post_process(response, chop_traces=True)
    print 'perturb'
    ## Add time shifts as given in the corrections filename
    write_container_to_dirs(convolved_traces, 'unperturbed', pad_traces=True)
    #print 'SKIPwrite'
    perturbation.apply(convolved_traces)
    write_container_to_dirs(convolved_traces, 'swarm_perturbed', pad_traces=True)

    # Save traces:
    #io.save(convolved_traces.traces_list(), 'swarm_stf.mseed')
    
    # Scrutinize the swarm using matplotlib
    #Visualizer(swarm, stats)
