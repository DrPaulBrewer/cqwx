#!/usr/bin/python
'''  cqwx.APT An APT receiver for NOAA Weather Satellites

     Copyright 2014 Paul Brewer KI6CQ
     License:  GNU General Public License version 3.0
     License:  http://www.gnu.org/copyleft/gpl.html
     NO WARRANTY. ALL USE OF THIS CODE IS AT THE SOLE RISK OF THE END USER.

     cqwx.APT is the APT python module for the cqwx python package 

     To use this APT receiver in python code:
        from cqwx.APT import RX
        rx = RX('/path/to/NOAA.wav')
        rx.fine_decode(pngfile='output.png')

     To use this APT receiver from the command line:

     python -m cqwx.APT NOAAaudioIN.wav roughDecode.png fineDecode.png
    
         cqwx.APT: literal.  this python module, the cqwx APT Receiver
         NOAAaudioIN.wav: placeholder for .wav file input
         roughDecode.png: placeholder for .png file output rough decode
         fineDecode.png:  placeholder for .png file output fine decode

'''
# APT.py Copyright 2014 Paul Brewer KI6CQ <drpaulbrewer AT gmail> 
#
# APT -- part of CQWX -- A weather satellite data decoder for NOAA APT format
#
import math
import sys
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

def _G5(x):
    """return copy of x as (len(x)/5) x 5 array, truncating remainder"""
    l = len(x)
    clip = 5*int(l/5)
    groupsOf5 = np.copy(x[0:clip])
    groupsOf5.shape=(clip/5,5)
    return groupsOf5

def _findPulseConvolve2(data,pulse,maxerr=0):
    """return indexes of locations of binary pulse in binary data
       data: binary 0-1 array of data to be searched
       pulse: binary 0-1 array
       maxerr: (optional) number of errors permitted"""
    required = len(pulse)-maxerr
    doubleConvolution = np.convolve(data,pulse[::-1])+\
                        np.convolve(1-data,1-pulse[::-1])
    return 1-len(pulse)+np.where(doubleConvolution>=required)[0]

class RX:
    
    """APT Receiver 

    decodes .wav files containing NOAA Weather Satellite audio into PNG files or raw data
    """
    
    def __init__(self, fname=None, signal = None):
        """Read the .wav file and initialize the receiver
        
        set either fname or signal, not both:
           fname (string): file name of .wav file
           signal (array): raw data of signal 
        
        .wav files must be recorded with a rate of 20800 samples/sec
        
        raises Exception if:
           neither fname nor signal is set
           supplied a .wav file is not at rate of 20800 samples/sec
           a file error occurs

        """ 
        self.rate = 20800 
        # we require 5 samples of the signal per APT data byte
        # 5 samples/byte*2040 bytes/line*2 APTlines/sec = 20400
        self.omega = 2*math.pi*2400.0/self.rate  
        # omega = subcarrier phase change per sample
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

    def _digitize(self, demodSig, plow=0.5, phigh=99.5):
        """ return 8 bit unsigned (0-255) digitized version of demodSig
        
        demodSig: signal input
        plow: percentile of demodSig to be asssigned 0 value
        phigh: percentile of demodSig to be assigned 255 value
        """
        (low, high) = np.percentile(demodSig, (plow, phigh))
        delta = high-low
        data = np.minimum(255, np.round(255*(demodSig-low)/delta))
        return data.astype(np.uint8)

    def _rough_demod(self):
        """ return root-mean-square demodulated signal

        requires no knowledge of the phase of the signal, location of sync
        pulses, etc.
        
        self.signal: input signal

        rms demodulator implementation summary
        -----------
        The input signal is grouped into groups of 5 samples.
        The sum of squares of each group of 5 samples is calculated
        This sum is multiplied by 2/5 and the square root returned.
        
        no side effects to self
        
        rms demodulator Theory Of Operation
        -------------------
        The rms demodulation method is roughly applicable to the NOAA APT
        data AM modulated in the carrier tone. As omega is about 41 degrees
        per sample, a group of 5 samples encompasses 205 degrees or about
        half a wave.  Squaring the samples amd summing has undesirable effects
        of summing squared noise, and also creating additional noise at 
        twice omega, but that is summing over roughly an entire wave, and an 
        entire wave usually sums neat zero.
        
        """
        return np.sqrt(2*np.sum(_G5(np.square(self.signal)), axis=1)/5)
        
    def rough_decode(self, pngfile=None):
        """ decode self.signal, create/return satellite image data
        
        image data will be created in self.rough_data

        pngfile: (optional) file name to output decoded PNG image from satellite
        
        The internal demodulator for this step is self._rough_demod.
        The self.signal is demodulated in chunks corresponding
        to a NOAA line length.  Each demodulated chunk is then digitized
        separately, localizing the effects of noise spikes.
        """
        
        for f in self.filters:
            f()
        raw = self._rough_demod()
        # do digitization line by line
        # to reduce impact of demodulated noise spikes on digitization
        self.rough_data = np.concatenate(\
            [self._digitize(ldata) \
             for ldata in np.split(raw, len(raw)/lenNOAAline)]\
        )

        hldata = 0+np.ravel(self.rough_data>127)
        self.As = _findPulseConvolve2(hldata,NOAAsyncA,1)
        self.A0 = None

        if len(self.As)>3:
            # if there are at least 3 syncAs
            # then calculate self.dAs, self.breaks, self.A0
            # and clip signal and rough_data to self.A0
            # sacrificing a portion of the final sec of data
            # to preserve full second alignment
            self.dAs = np.diff(self.As)
            self.breaks = (self.dAs % lenNOAAline) !=0        
            self.A0 = self.As[0] % lenNOAAline
            self.As = self.As - self.A0
            self.duration = self.duration - 1
            self.signal = self.signal[(5*self.A0):(5*self.A0+self.rate*self.duration)]
            self.rough_data = self.rough_data[self.A0:(-2*lenNOAAline+self.A0)]

        if len(self.rough_data.shape)==1:
            self.rough_data.shape = \
                (len(self.rough_data)/lenNOAAline, lenNOAAline)

        if pngfile is not None:
            self.makePNG(pngfile, 'rough')
        return self.rough_data

    def _findPAS(self, start, end):
        """return (estimate of phase, amplitude, standard error of amplitude)
        
        start: index to start of NOAA data line in self.signal
        end: index to end of NOAA data line in self.signal
 
        For proper operation, start must point to the signal index at the
        beginning of a line of data, i.e. at the beginning of a syncA pulse.  
       
        Determine the phase of the NOAA APT 2400 hz carrier tone using a
        portion of tbe signal called SPACE A.  SPACE A modulates the carrier
        at a constant level corresponding to white or black.  
       
        """
        
        offset = 5*lenNOAAsyncA
        length = 5*lenNOAAspace
        demodAM = self._fine_demod
        if (end-start) < (offset+length):
            raise(Exception("findPAS: signal is of insufficient length"))
        M = [ np.mean(demodAM(start+offset,start+offset+length,phase)) \
                  for phase in -3.14+0.01*np.arange(628) ]
        phase = -3.14+0.01*np.argmax(M)
        amplitude = np.max(M)
        return (phase, amplitude, np.std(demodAM(start+offset,start+offset+length,phase)))

    def _findJPA(self, start, end):
        demodAM = self._fine_demod
        j0 = 0
        j1 = 0
        if start>2:
            j0=-2
        if (end+2)<len(self.signal):
            j1=2
        J = j0 + np.arange(1+j1-j0)
        PAS = [ self._findPAS(start+j,end+j) for j in J]
        V = [ np.var(demodAM(start+j, start+j+5*lenNOAAsyncA, PAS[i][0])) \
              for (i,j) in enumerate(J) ]
        argmaxV = np.argmax(V)
        return (J[argmaxV], PAS[argmaxV][0], PAS[argmaxV][1])

    def _fine_demod(self, start, end, phase):
        """ least squares demodulation of signal against reference signal
        
        input signal to demodulator is self.signal[start:end]
        start: index where demodulation begins
        end:  index where demodulatione ends
        phase: phase of 2400 Hz reference carrier to apply at self.signal[0]
        
        """
        sIN = self.signal[start:end]
        l = len(sIN)
        adj = l%5
        s = np.copy(sIN[0:(l-adj)])
        g = np.cos((start+np.arange(len(s)))*self.omega+phase)
        g25 = _G5(g*g)
        invsumsq = np.repeat(np.reciprocal(np.sum(g25, axis=1)), 5)
        out = np.maximum(0, np.sum(_G5(invsumsq*g*s), axis=1))
        return out

    def _residual(self, start, end, phase, demod):
        """ determine residual signal from demod amplitudes and phase
        
        input signals are self.signal[start:end] and
        a signal synthesiszed from demod, phase
        """
        l = end-start
        local_osc = np.cos((start+np.arange(l))*self.omega+phase)
        synth = local_osc*np.repeat(demod,5)
        return self.signal[start:end]-synth
        
    def fine_decode(self, repair=True, dejitter=True, pngfile=None, residuals=False):
        """ 
        extensive decoding of self.signal beginning with finding the 
        sync pulse in self.rough_data and using the pulse locations to
        guide the estimation of signal phase and demodulation with a 
        least-squares method in self._fine_demod
        
        create self.fine_data containing satellite image data and returns
        data to caller.  Generate pngfile if requested.
                
        rows of self.fine_data are synchronized to satellite sync-A pulses

        create diagnostic arrays with one entry per decoded data line:
           self.fine_demod_phase:  phase of 2400hz carrier, from space A
               The reference time of this phase is self.signal[0]
           self.fine_demod_jitter: if dejitter, the index offset, else 0
           self.fine_demod_space_amp:  mean amplitude of the carrier in space A
        
        repair: (True) replace missing/mis-synced line with average of lines above and below
        dejitter: (False) test and correct for misalignment of signal samples 
        once at each line, using goodness-of-fit of sync pulses (TO DO)
        
        pngfile: (optional) file name to output decoded PNG image from satellite
        
        """
        if self.rough_data is None:
            self.rough_decode()
        # Use rough data to find syncA pulses 
        # and align fine demodulation efforts around these pulses.
        # If pulses show up in odd places, mark line and postprocess.
        # The known portions of the signal (syncA, spaceA) are used to estimate
        # phase and sample jitter for the demodulator.
        triplets = np.concatenate( (self.As[0:-1], self.dAs, self.breaks) )
        triplets.shape=(3, len(self.dAs))
        lineData = []
        skipIdx = []
        self.fine_demod_jitter = []
        self.fine_demod_phase = []
        self.fine_demod_space_amp = []
        if residuals:
            self.residuals = []
        else:
            self.residuals = None
        for (idx, delta, skipline) in triplets.T:
            for k in range(0,delta/lenNOAAline):
                start = 5*idx+5*k*lenNOAAline
                end = start+5*lenNOAAline
                if dejitter:
                    (jitter, phase, amp)  = self._findJPA(start,end)
                else:
                    jitter = 0
                    (phase, amp, sdev) = self._findPAS(start,end)
                self.fine_demod_jitter.append(jitter)
                self.fine_demod_phase.append(phase)
                self.fine_demod_space_amp.append(amp)
                raw = self._fine_demod(start+jitter,\
                                        end+jitter,\
                                        phase)
                if residuals:
                    self.residuals.append(self._residual(start+jitter,\
                                                         end+jitter,\
                                                         phase,\
                                                         raw))
                line = 255-self._digitize(raw)
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
        if type(residuals)==type('string'):
            r = np.array(self.residuals)
            r.shape = r.shape[0]*r.shape[1]
            scipy.io.wavfile.write(residuals,\
                                   self.rate,\
                                   np.round(r).astype('int16'))
        return self.fine_data


    def _dcfilter(self):
        """modify self.signal to filter out DC-1Hz
        subtracts mean from each one second grouping of self.signal
                
        return 1Hz DC observations 
        
        """
        oneSecondDC = np.fromiter( (np.mean(s) for s in np.split(self.signal, self.duration)), np.float)
        self.signal = self.signal - np.repeat(oneSecondDC,self.rate)
        self.DC = oneSecondDC
        return self.DC

    def _quickpopfilter(self):
        """modify self.signal to filter out large positive/negative values
        
        clips the entire self.signal to the max and min of first 100,000 samples 
        """
        if len(self.signal) > 100000:
            max100000 = max(self.signal[0:100000])
            self.signal[self.signal>max100000]=max100000
            min100000 = min(self.signal[0:100000])
            self.signal[self.signal<min100000]=min100000

    def freq_counter(self):
        """return one-second estimates of freq of self.signal
        uses zero-crossings method
        
        Should approach 2400 Hz in strong signal to noise environment.
        """
        return [ np.sum(np.abs(np.diff(np.sign(s-np.mean(s))))/4) \
                 for s in np.split(self.signal, self.duration) ]

    def makePNG(self, fname, datasource):
        """create PNG file from decoded satellite image data
        
        fname:  file name to use in creating the png file.
        datasource:  'rough' -- use self.rough_data for image
                     'fine'  -- use self.fine_data for image
        
        """
        data = None
        if datasource == 'rough':
            data = np.copy(self.rough_data)
        if datasource == 'fine':
            data = np.copy(self.fine_data)
        if data is None:
            raise Exception("Fatal: makePNG requires datasource='fine' or 'rough'")
        im = PIL.Image.fromarray(data)
        im.save(fname,'PNG')
        


def _NOAA_test_signal(minmod=0.05,maxmod=0.95,phase=0,domega=0,e=0,data=None):
    """return simulated NOAA signal with syncA pulse, space, and random data
    
    return length is 10200 byytes, equivalent to one NOAA line
    
    does not simulate syncB, spaceB, or telemetry bars.
    
    minmod: (0.05) minimum modulation level of 2400 hz carrier
    maxmod: (0.95) maximum modulation level of 2400 hz carrier
    phase: (0.0) phase of 2400 hz carrier at beginning of line
    domega: (0.0) difference in omega representing off-frequency carrier
    e: (0.0) standard deviation (signma) of Gaussian noise
    data: (if data is None) default to generating random data
          data may be specified for the non-syncA, non-space content
    
    """
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


if __name__== "__main__":
    if len(sys.argv)<3:
        print __doc__
    rx = RX(sys.argv[1])
    if len(sys.argv)>=3:
        rx.rough_decode(pngfile=sys.argv[2])
    if len(sys.argv)>=4:
        rx.fine_decode(pngfile=sys.argv[3])




