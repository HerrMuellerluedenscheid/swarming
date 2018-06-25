from pyrocko import trace, pz, io, pile, util
import glob
import logging
import numpy as num
import os

pjoin = os.path.join
logger = logging.getLogger()


def load_station_system_mapping(fn):
    mapping = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            k, v = line.split()
            mapping[k] = v
    return mapping


def apply_response(target, tr, pz, code_mapping, transfer_dict):
    k = code_mapping[target.codes]
    response = pz[k]
    transfer_dict.update({'transfer_function': response})
    return tr.transfer(**transfer_dict)


def load_poles_zeros(super_dir='', fns=''):
    pzs = {}
    for fn in fns:
        zeros, poles, constant = pz.read_sac_zpk(pjoin(super_dir, fn))
        #if quantity=='displacement':
        #    print 'WARNING: The simulation needs to be checked!'
        #    zeros.append(0.j)\
        # remove one zero -> differentiate
        #zeros.pop()
        #zeros.append(0.0j)
        response = trace.PoleZeroResponse(zeros, poles, num.complex(constant))
        pzs[fn] = response
    return pzs


class TargetCodeMapper:
    def __init__(self, broad_bands):
        self.broad_bands = broad_bands

    def __getitem__(self, codes):
        if codes[1] in self.broad_bands:
            key = 'guralp_cmg3_120s'
        else:
            key = 'mark-L4C-3D'
        return key


class Noise():
    def __init__(self, files, scale=1., station_mapping={}, network_mapping={},
                 channel_mapping={}, location_mapping={}):
        self.scale = scale
        self.nslc_mapping = {}
        self.data_pile = pile.make_pile(files)
        self.noise = self.dictify_noise(self.data_pile, network_mapping, channel_mapping,
                                        station_mapping, location_mapping)
        self.merge_fader = trace.CosFader(xfrac=0.1)

    def dictify_noise(self, data_pile, network_mapping={}, channel_mapping={}, station_mapping={}, location_mapping={}):
        noise = {}
        for tr in data_pile.iter_traces(load_data=True):
            if tr.network in network_mapping:
                n = network_mapping[tr.network]
            else:
                n = tr.network
            if tr.station in station_mapping:
                s = station_mapping[tr.station]
            else:
                s = tr.station
            if tr.channel in channel_mapping:
                c = channel_mapping[tr.channel]
            else:
                c = tr.channel
            if tr.location in location_mapping:
                l = location_mapping[tr.location]
            else:
                l = tr.location
            key = (n, s, l, c)
            if not key in noise.keys():
                noise[key] = tr.copy(data=True)
            else:
                logger.warn('More than one noisy traces for %s' % ('.'.join(tr.nslc_id)))
        return noise

    def noisify(self, tr):
        if tr.station not in self.noise.keys():
            raise Exception('No Noise for tr %s' % ('.'.join(tr.nslc_id)))
        n = self.extract_noise(tr) * self.scale
        tr.ydata += n
        return tr

    def extract_noise(self, tr):
        n_want = len(tr.ydata)
        i_start = num.random.choice(range(len(self.noise[tr.station])-n_want))
        return self.noise[tr.station][i_start:i_start+n_want]

    def make_noise_trace(
            self, tmin, tmax, nslc_id, target_nslc_id=None, merge_traces=None,
            outdir='noise_concat', template='tr_%n.%s.%l.%c.-%(tmin)s.mseed'):

        if target_nslc_id is None:
            target_nslc_id = nslc_id

        n, s, l, c = target_nslc_id
        noise_tr = self.noise[(n, s, l, c)]
        deltat = noise_tr.deltat
        self.resample_many(merge_traces, deltat)
        overlap = 0.2                           # 20 percent of sample length
        sample_length = 60.                     # seconds
        ns = int(sample_length/deltat)          # number of samples
        fader = trace.CosFader(xfrac=overlap)
        noisey = noise_tr.get_ydata()
        buffer_size = num.int(sample_length*60./deltat)
        i_dumped = 0.
        buffer_y = num.zeros(buffer_size)
        ioverflow = 0
        overflow = None
        taper = trace.costaper(
            0., overlap/2.*sample_length, sample_length*(1.0-overlap/2.),
            sample_length, ns, deltat)
        for i, istart in enumerate(num.arange(0, int(num.ceil((tmax-tmin)/deltat)), int(ns*(1.0 - overlap/2.0)))):
            istart = int(num.floor((istart+ioverflow) % buffer_size))
            istop = istart + ns
            nmissing = buffer_size - istop
            isample_start = int(num.floor(num.random.uniform(0, len(noisey)-ns)))
            isample_stop = isample_start + ns
            noise_sample = num.zeros(ns)
            noise_sample[:] = noisey[isample_start:isample_stop]
            noise_sample *= taper
            if nmissing<0:
                isample_stop = ns + nmissing
                istop = istart + ns + nmissing
                ioverflow = isample_stop
                overflow = noise_sample[ioverflow:]
                noise_sample = noise_sample[:ioverflow]
            buffer_y[istart: istop] += noise_sample
            # Problem mit ioverlap.
            if overflow is not None:
                tmin_tr = tmin+i_dumped*buffer_size*deltat
                tmax_tr = tmin_tr+(buffer_size-1)*deltat
                buff_tr = trace.Trace(network=n, station=s, location=l, channel=c,
                                      tmin=tmin_tr, tmax=tmax_tr,
                                      deltat=noise_tr.deltat, ydata=buffer_y)
                if merge_traces is not None:
                    for mtr in merge_traces:
                        if buff_tr.is_relevant(mtr.tmin, mtr.tmax):
                            mtr = mtr.taper(self.merge_fader, inplace=False)
                            buff_tr.add(mtr)
                            #print max(num.abs(mtr.ydata))
                            #trace.snuffle([mtr, buff_tr])
                io.save(buff_tr, pjoin(outdir, template))
                buffer_y = num.zeros(buffer_size)
                buffer_y[0:ns-ioverflow] = overflow
                i_dumped += 1.
                overflow = None

    def resample_many(self, traces, deltat):
        for tr in traces:
            if num.abs(deltat-tr.deltat)>1e-7:
                tr.resample(deltat*(1.+1e-4))


    def __str__(self):
        s = ''
        for k, v in self.noise.items():
            s += "%s %s\n"%(k,v)
        return s

