from imports import *
from scipy.io.wavfile import read as readwav
from numpy.lib.stride_tricks import as_strided

sample_rate = 48000             # sample rate for samples and input
def load_wavs_from_dir(path):
    fnames = map(lambda f: f.path,
                 filter(
                     lambda f: f.is_file(),
                     os.scandir(path)))
    fnames, rates, wavs = map(np.asarray,
                              zip(*tuple(
                                  map(
                                      lambda f: (f, *readwav(f)),
                                      fnames))
                              )
    )
    wrong_rates = (rates != sample_rate)
    if wrong_rates.any():
        print(
            'Following files have wrong simple rate (!= {}):'.format(sample_rate) +
            '\n\t' + '\n\t'.join(
                map(lambda t: '"{}" | rate = {}'.format(*t),
                    np.vstack((fnames, rates))[:,wrong_rates].transpose()
                )),
            file=stderr)
        print('\nSkipping them')

    wavs = wavs[~wrong_rates]

    maxlen = max(i.shape[0] for i in wavs)
    ar = []
    for i in wavs:
        s = i[:] if len(i.shape) == 1 else i[:,0] # Use single channel
        s = s.copy()
        s.resize((maxlen,))
        ar.append(s)

    data = np.vstack(ar)
    return data

def record_samples(nil):
    return

def stft(x, fftsize=64, overlap=.5):   
    hop = int(fftsize * (1 - overlap))
    w = sp.hanning(fftsize + 1)[:-1]    
    raw = np.array([
        np.fft.rfft(w * x[i:i + fftsize])
        for i in range(0, len(x) - fftsize, hop)
    ])
    return raw[:, :(fftsize // 2)]



# Find frequency peaks in data
# http://kkjkok.blogspot.com/2013/12/dsp-snippets_9.html 
def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
    win_size = l_size + r_size + c_size
    shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
    strides = x.strides + (x.strides[-1],)
    xs = as_strided(x, shape=shape, strides=strides)

    def is_peak(x):
        centered = (np.argmax(x) == l_size + int(c_size/2))
        l = x[:l_size]
        c = x[l_size:l_size + c_size]
        r = x[-r_size:]
        passes = np.max(c) > np.max([f(l), f(r)])
        if centered and passes:
            return np.max(c)
        else:
            return -1

    r = np.apply_along_axis(is_peak, 1, xs)
    top = np.argsort(r, None)[::-1]
    heights = r[top[:n_peaks]]
    #Add l_size and half - 1 of center size to get to actual peak location
    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
    return heights, top[:n_peaks]



