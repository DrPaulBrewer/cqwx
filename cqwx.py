# cqwx.py Copyright 2014 Paul Brewer KI6CQ <drpaulbrewer@gmail.com> 
#
# CQWX -- A weather satellite data decoder for NOAA APT format
#
# License: GNU General Public License v3.0
# 
# NO WARRANTY.  ALL USE OF THIS CODE IS AT THE SOLE RISK OF THE END USER.
#
# You should have received a copy of the LICENSE with this file.
# If not, see http://www.gnu.org/copyleft/gpl.html
#
# The GNU General Public License v3.0 does NOT permit distributing devices
# that run this code, unless those devices include full source code that
# the end user can modify. Required source includes not only this code but 
# also any code that calls this code.    
# 
# For such cases, a commercial LICENSE can be purchased from the author
# for a reasonable fee.  Contact drpaulbrewer@gmail.com
#
# Contributions of code or funding towards improvement of this code
# are appreciated. 
#
#
import math
import numpy as np
import scipy.io.wavfile
import PIL

# these are the NOAA parameters for NOAA APT 
# transmitted by NOAA-15, NOAA-18, and NOAA-19 satellites
NOAAsyncA = np.concatenate( ([4*[0],7*[1,1,0,0],7*[0]]) )
lNOAAsyncA = np.concatenate( (4*[11], 7*[244,244,11,11], 7*[11]) )
lenNOAAsyncA = len(NOAAsyncA)
lenNOAAspace = 47
lenNOAAline = 2080

def G5(x):
    l = len(x)
    clip = 5*int(l/5)
    groupsOf5 = np.copy(x[0:clip])
    groupsOf5.shape=(clip/5,5)
    return groupsOf5

def findPulseConvolve2(data,pulse,maxerr=0):
    required = len(pulse)-maxerr
    doubleConvolution = np.convolve(data,pulse[::-1])+\
                        np.convolve(1-data,1-pulse[::-1])
    return 1-len(pulse)+np.where(doubleConvolution>=required)[0]

class APTReceiver:
    
    rate = 20800   # we require 5 samples of the signal per APT data byte
    omega = 2*math.pi*2400.0/rate  # omega = subcarrier phase change per sample


    def __init__(self, fname=None, signal = None):
        if signal is not None:
            self.signal = signal
        else:
            if fname is None:
                raise Exception("Fatal: cqwx.APTReceiver.__init__() requires signal source")
            else:
                (wavrate, self.signal) = scipy.io.wavfile.read(fname)
                if wavrate!=self.rate:
                    raise Exception("Fatal: Need wav file rate"+str(self.rate)+". Got rate="+str(wavrate)+". Adjust SDR sample rate or perhaps use a utility like sox to resample.")
                # truncate signal at last one second sample
                # this makes sure we have an integer number of seconds 
                # and also an even number of APT lines
                trunc = self.rate*int(len(self.signal)/self.rate)
                self.signal = self.signal[0:trunc]
        # for all signals 
        self.duration = len(self.signal)/self.rate  # set int sec duration
        self.filters = [self._quickpopfilter, self._dcfilter] # set filter order
        self.rough_data = None

    def digitize(self, demodSig, plow=0.5, phigh=99.5):
        (low, high) = np.percentile(demodSig, (plow, phigh))
        delta = high-low
        data = np.minimum(255, np.round(255*(demodSig-low)/delta))
        return data.astype(np.uint8)

    def rough_demod(self):
        return np.sqrt(2*np.sum(G5(np.square(self.signal)), axis=1)/5)
        
    def rough_decode(self, pngfile=None):
        for f in self.filters:
            f()
        raw = self.rough_demod()
        # do digitization line by line
        # to reduce impact of demodulated noise spikes on digitization
        self.rough_data = np.concatenate(\
            [self.digitize(ldata) for ldata in np.split(raw, len(raw)/lenNOAAline)]\
        )
        # flatten the data array
        self.rough_data.shape=len(raw)
        if pngfile is not None:
            self.makePNG(pngfile, 'rough')
        return self.rough_data

    def _findPAS(self, start, end):
        # global lenNOAAsyncA, lenNOAAspace, omega
        offset = 5*lenNOAAsyncA
        length = 5*lenNOAAspace
        demodAM = self.fine_demod
        if (end-start) < (offset+length):
            raise(Exception("findPAS: signal is of insufficient length"))
        M = [ np.mean(demodAM(start+offset,start+offset+length,phase)) \
                  for phase in -3.14+0.01*np.arange(628) ]
        phaseAtOffset = -3.14+0.01*np.argmax(M)
        amplitude = np.max(M)
        phase = ( phaseAtOffset - self.omega*offset ) % (2*math.pi)
        return (phase, amplitude, np.std(demodAM(start+offset,start+offset+length,phaseAtOffset)))

    def fine_demod(self, start, end, phase):
        sIN = self.signal[start:end]
        # global omega
        l = len(sIN)
        adj = l%5
        s = np.copy(sIN[0:(l-adj)])
        g = np.cos(np.arange(len(s))*self.omega+phase)
        g25 = G5(g*g)
        invsumsq = np.repeat(np.reciprocal(np.sum(g25, axis=1)), 5)
        out = np.maximum(0, np.sum(G5(invsumsq*g*s), axis=1))
        return out
        
    def fine_decode(self, repair=True, dejitter=False, pngfile=None):
        if self.rough_data is None:
            self.rough_decode()
        # use rough data to find syncA pulses 
        # and align fine demodulation efforts around these pulses
        # if pulses show up in odd places, mark line and postprocess
        # the known portions of the signal can then be used to estimate
        # phase and sample jitter
        As = findPulseConvolve2(self.rough_data>127,NOAAsyncA,1)
        dAs = np.diff(As)
        breaks = (dAs % lenNOAAline) !=0
        triplets = np.concatenate( (As[0:-1], dAs, breaks) )
        triplets.shape=(3, len(dAs))
        lineData = []
        skipIdx = []
        if dejitter is True:
            raise Exception("Fatal: dejitter=True: no dejitter code yet")
        for (idx, delta, skipline) in triplets.T:
            for k in range(0,delta/lenNOAAline):
                start = 5*idx+5*k*lenNOAAline
                end = start+5*lenNOAAline
                phase = self._findPAS(start,end)[0]
                line = 255-self.digitize(\
                        self.fine_demod(start,\
                                        end,\
                                        phase)\
                    )
                lineData.append(line)
            if skipline:
                lineData.append(127+np.zeros(lenNOAAline, dtype='uint8'))
                skipIdx.append(len(lineData)-1)
        if repair:
            for idx in skipIdx:
                if (idx<len(lineData)-2):
                    lineData[idx] = lineData[idx-1]/2+lineData[idx+1]/2
        self.fine_data = np.array(lineData)
        if pngfile is not None:
            self.makePNG(pngfile, 'fine')
        return self.fine_data


    def _dcfilter(self):
        oneSecondDC = np.fromiter( (np.mean(s) for s in np.split(self.signal, self.duration)), np.float)
        self.signal = self.signal - np.repeat(oneSecondDC,self.rate)
        self.DC = oneSecondDC
        return self.DC

    def _quickpopfilter(self):
        if len(self.signal) > 100000:
            max100000 = max(self.signal[0:100000])
            self.signal[self.signal>max100000]=max100000
            min100000 = min(self.signal[0:100000])
            self.signal[self.signal<min100000]=min100000

    def freq_counter(self):
        return [ np.sum(np.abs(np.diff(np.sign(s-np.mean(s))))/2) \
                 for s in np.split(self.signal, self.duration) ]

    def makePNG(self, fname, datasource):
        data = None
        if datasource == 'rough':
            data = np.copy(self.rough_data)
            data.shape = (len(data)/lenNOAAline, lenNOAAline)
        if datasource == 'fine':
            data = np.copy(self.fine_data)
        if data is None:
            raise Exception("Fatal: makePNG requires datasource='fine' or 'rough'")
        im = PIL.Image.fromarray(data)
        im.save(fname,'PNG')
        


def NOAA_test_signal(minmod=0.05,maxmod=0.95,phase=0,domega=0,e=0,data=None):
    # D to A = min+(max-min)*d/255.0
    space = np.repeat(255*np.random.random_integers(0,1,1), lenNOAAspace)
    ldata = lenNOAAline-len(lNOAAsyncA)-lenNOAAspace
    if data is None:
        message = np.random.random_integers(0,255,ldata)
    else:
        if len(data)!=ldata:
            raise(Exception("requires data of length "+str(ldata)))
        message = data
    linedata = np.concatenate( (lNOAAsyncA, space, message) )
    modulation = minmod+(maxmod-minmod)*np.repeat(linedata,5)/255.0
    signal = modulation*np.cos((omega+domega)*np.arange(5*2040)+phase)+\
        e*np.random.normal(0.0,1.0,5*2040)
    return (signal, linedata)


