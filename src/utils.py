from pyrocko.gf.seismosizer import Target
from pyrocko.util import match_nslc


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


def get_targets(stations, data_pile, store_id=None):
    targets = []
    nslc_ids = data_pile.nslc_ids
    for station in stations:
        for nslc_id in nslc_ids:
            if match_nslc('%s.*' % station.nsl_string(), nslc_id):
                targets.append(
                    Target(
                        codes=nslc_id,
                        lat=station.lat,
                        lon=station.lon,
                        elevation=station.elevation,
                        depth=station.depth,
                        store_id=store_id)
                    )

                nslc_ids.remove(nslc_id)


    return targets
