import numpy as num
import os
import progressbar
import logging
import sys

from pyrocko.gf.seismosizer import RemoteEngine, Target, LocalEngine, Request, Response
from pyrocko.model import load_stations, dump_events
from pyrocko import io, trace
from pyrocko.guts import Object, String, Int, Float
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


def iter_result_by_sources(response):
    for isource, source in enumerate(response.request.sources):
        yield source, response.results_list(isource)


def get_variable(key):
    try:
        return os.environ[key]
    except KeyError as e:
        print('No environ variable %s assigned' % key)
        sys.exit()

def setup():

    apply_corrections = False

    # I use the following environment variables to locate green function 
    # stores and the station files, which in this case are the webnet stations.

    webnet = get_variable("WEBNET")

    # Number of sources....
    number_sources = 100

    # swarm geometry.
    # center_lat, center_lon and center_depth define the center point of
    # the desired swarm area.
    # The orientation is then given as follows.
    # azimuth is the angle versus north (rotation around z-axis), dip is the
    # the rotation around the horizontal y axis and tilt around x axis.
    # length, depth and thickness describe the x-, z- and y-extension
    geometry = CuboidSourceGeometry(center_lon=12.45, 
                                     center_lat=50.214,
                                     center_depth=10000,                                     #dip=-75.,
                                     azimuth=170,
                                     dip=80.,
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
    # one_day =120
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
                                   variation=5)

    # magnitude distribution with a- and b- value and a minimum magnitude.
    magnitudes = MagnitudeDistribution.GutenbergRichter(a=1, b=0.5, Mmin=0.08)

    # The store we are going extract green functions from:
    store_id = 'qplayground_total_4_mr_full'
    engine = LocalEngine(use_config=True,
                         store_superdirs=['/data/stores'],
                         default_store_id=store_id)

    try:
        store = engine.get_store()
        config = store.config
        model = config.earthmodel_1d
    except OSError:
        print('passing OSError. No Model assigned.')
        pass

    # Gather these information to create the swarm:
    s = Swarm(geometry=geometry,
                  timing=timing,
                  mechanisms=mechanisms,
                  magnitudes=magnitudes)

    dump_events(s.get_events(), 'events_swarm.pf')

    return s


class DataProducer(Object):
    fn_stations = String.T()
    quantity = String.T(default='velocity')

    def process_swarm(swarm, skip_post_process=False):
        stats = load_stations(self.fn_stations)

        # convert loaded stations to targets (see function at the top).
        targets = guess_targets_from_stations(stats, quantity='velocity')
        logger.info('processing request...')

        # Processing that data will return a pyrocko.gf.seismosizer.Reponse object.
        response = engine.process(
            sources=swarm.get_sources(),
            targets=targets)

        #logger.info('done')
        #io.save(response.pyrocko_traces(), 'swarm.mseed')
        # convolved_traces = stf.post_process(response, chop_traces=True)
        ## Add time shifts as given in the corrections filename
        #write_container_to_dirs(convolved_traces, 'unperturbed', pad_traces=True)
        # corrections = os.path.join(os.environ['HOME'],
        #                            'src/seismerize',
        #                            'residuals_median_CakeResiduals.dat')

        # Add a perturbation. *revert=True* means, that the traces are shifted by the correction time * -1.
        # perturbation = Perturbation.from_file(corrections, revert=True)
        # perturbation.apply(convolved_traces)

        noise = GaussNoise(degap=True, mu=0.00000001)
        noised = noise.apply(convolved_traces)

        return iter_result_by_sources(response)

        # trace.snuffle(noised.values(), events=events)

        #write_container_to_dirs(convolved_traces, 'swarm_perturbed', pad_traces=True)

        # Save traces:
        # io.save(convolved_traces.traces_list(), 'swarm_stf.mseed')

        # Scrutinize the swarm using matplotlib
        # Visualizer(swarm, stats)


if __name__ == '__main__':
    swarm = setup()
    for s, trs in process_swarm(swarm, skip_post_process=True):
        print(s, trs)