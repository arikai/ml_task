import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FuncAnimation
import sounddevice as sd
import numpy as np
from math import ceil

class TapePlotter(object):
    """
    Plot microphone output as it is recorded
    """

    def __init__(self, device=None, samplerate=48000, length=10, fps=60):
        self._samplerate = samplerate
        self._channels = 1
        self._device = device
        self._tape_duration = length # seconds

        self._interval = 1.0/fps
        self._blocksize = int(self._samplerate * self._interval) # Number of frames passed to callback

        # No need to plot all the data: select mean of every n measurements
        self._step_size = 15 * self._tape_duration # Number of measurements used to calculate average
        self._plot_blocksize = int(self._blocksize / self._step_size)

        plot_blocks_num = int(self._samplerate * self._tape_duration / self._step_size)
        self._tape = np.empty(plot_blocks_num)*np.nan
        self._time_axis = np.linspace(0, self._tape_duration, plot_blocks_num)

        self._fig, self._ax = plt.subplots()
        self._ax.set_xlim(0, self._tape_duration)
        self._ax.set_ylim(-1.5, 1.5)

        self._linestyle = { 'color': 'b', 'linestyle': '-' }
        self._markstyle = { 'color': 'g' }

        self._line = plt.Line2D((), (), **self._linestyle)
        self._marks = []
        # self._marks_timestamps = []

        self._ax.add_line(self._line)

        self._update_callbacks = []

    def __stream_callback(self, indata, frames, timestamp, status):
        plot_blocks = (indata[:self._plot_blocksize * self._step_size] 
                      .reshape((self._plot_blocksize, self._step_size))
                      .mean(1))
        self.__tape_add(plot_blocks)
        for cb in self._update_callbacks:
            cb(indata, self.tape())

    def __tape_add(self, block):
        if np.abs(block).mean() > 0.8 \
           and (len(self._marks) == 0 or self._marks[-1].get_x() + 0.5 < self._tape_duration):
            ylims = self._ax.get_ylim()
            mark = plt.Rectangle(
                (self._tape_duration - self._interval, ylims[0]),
                0.5, ylims[1] - ylims[0],
                **self._markstyle
            )
            self._ax.add_patch(mark)
            self._marks.append(mark)
            print("mark added")

        i = 0
        while i < len(self._marks):
            mark = self._marks[i]
            x = mark.get_x() - self._interval * 0.81
            if x + mark.get_width() <= 0:
                mark.remove()
                self._marks.pop(0)
                print("mark removed")
            else:
                mark.set_x(x)
                i += 1
        # print(self._marks)

        self._tape[:-self._plot_blocksize] = self._tape[self._plot_blocksize:]
        self._tape[-self._plot_blocksize:] = block

    def __animate_callback(self, frame):
        self._line.set_data(self._time_axis, self._tape)
        return (*self._marks, self._line)

    def add_callback(self, callback):
        self._update_callbacks.append(callback)
        
    def tape(self):
        return self._tape
    
    def plot(self):
       with sd.InputStream(
               samplerate = self._samplerate,
               blocksize  = self._blocksize,
               channels   = self._channels,
               dtype      = np.float32,
               device     = self._device,
               callback   = self.__stream_callback,
       ):
           ani = FuncAnimation(
               self._fig, self.__animate_callback,
               interval = int(1000*self._interval), # ~ 60 FPS
               blit = True
           )
           plt.show(block=True)
