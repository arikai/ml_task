#!/usr/bin/env python3

import os
import sys
from sys import stderr
import numpy as np
from sklearn import svm
from pprint import pprint
from scipy.io.wavfile import read as readwav

from tape_plotter import TapePlotter

# for some reason, input doesn't work properly on my system
_input = input
def input(prompt=''):
    print(prompt, end=' ')
    return _input()

sample_rate = 48000             # sample rate for samples and input
def menu():
    def record_samples(nil):
        return

    def process_samples(direntry):
        filter(lambda f: f.is_file(), os.scandir(direntry.path))
        fnames = map(lambda f: f.path,
                     filter(
                         lambda f: f.is_file(),
                         os.scandir(direntry.path)))
        fnames, rates, wavs = map(np.asarray,
                                  zip(*tuple(
                                      map(
                                          lambda f: (f, *readwav(f)),
                                          fnames))
                                      )
                                  )
        wrong_rates = (rates != sample_rate)
        if wrong_rates.any():
            print('Following files have wrong simple rate (!= {}):'.format(sample_rate),
                file=stderr)
            print('\n\t', end='')
            print('\n\t'.join(map(lambda t: '"{}" | rate = {}'.format(*t),
                np.vstack((fnames, rates))[:,wrong_rates].transpose()
            )))
            print('\nSkipping them')

        wavs = wavs[~wrong_rates]
        for i in wavs:
            print(len(i.shape), i[:,0].shape)

        maxlen = max(i.shape[0] for i in wavs)
        ar = []
        for i in wavs:
            s = i[:] if len(i.shape) == 1 else i[:,0]
            s = s.copy()
            s.resize((maxlen,))
            ar.append(s)

        data = np.vstack(ar)
        return data

    base_dir = os.path.dirname(os.path.realpath(__file__))
    samples_dir = base_dir + '/samples'

    menu_entries = []

    if os.path.isdir(samples_dir):
        os.chdir(samples_dir)
        menu_entries.extend(
            ((d.name, d, process_samples)
            for d in filter(lambda f: f.is_dir(), os.scandir())
            )
        )

    menu_entries.extend((
        ('Record samples', (), record_samples),
        ('Quit', (), lambda _: sys.exit(0))
    ))

    for i in range(len(menu_entries)):
        print('{}. {}'.format(i+1, menu_entries[i][0]))
    print()

    while True:
        try:
            c = int(input('Choose a pattern to recognize (1-{}):'
                        .format(len(menu_entries))))
            if c > len(menu_entries):
                raise ValueError()
            c -= 1
            break
        except ValueError:
            print('A number must be typed (1-{})'
                .format(len(menu_entries)))

    entry = menu_entries[c]
    train_data = entry[2](entry[1])

if __name__ == '__main__':
    data = menu()

    clf = svm.SVR(data)
    tp = TapePlotter(length=5, fps=60)
    tp.plot()

