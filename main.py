#!/usr/bin/env python3

import sys
from sklearn import svm
import scipy.signal as sig

from imports import *
from tape_plotter import TapePlotter
from waves import load_wavs_from_dir, record_samples

# for some reason, input doesn't work properly on my system
_input = input
def input(prompt=''):
    print(prompt, end=' ')
    return _input()

def menu():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    samples_dir = base_dir + '/samples'

    menu_entries = []

    if os.path.isdir(samples_dir):
        os.chdir(samples_dir)
        menu_entries.extend(
            ((d.name, d.path, load_wavs_from_dir)
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
    return train_data

if __name__ == '__main__':
    data = menu()               # [wav1, wav2, wav3, ...]
    # map(
    #     lambda wav: sig.stft(wav, fs=sample_rate, return_onesided=True),
    #     data
    # )
    
    clf = svm.SVC(probability=True)
    # clf.fit(data, np.arange(0, len(data[0])))
    tp = TapePlotter(length=1, fps=60)
    tp.plot()
