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
import scipy.signal
import PIL

# these are the NOAA parameters for NOAA APT 
# transmitted by NOAA-15, NOAA-18, and NOAA-19 satellites
NOAAsyncA = np.concatenate( ([4*[0],7*[1,1,0,0],7*[0]]) )
lNOAAsyncA = np.concatenate( (4*[11], 7*[244,244,11,11], 7*[11]) )
lNOAAsyncB = np.concatenate( (4*[11], 7*[244,244,244,11,11]) )
lenNOAAsyncA = len(NOAAsyncA)
lenNOAAspace = 47
lenNOAAimage = 909
lenNOAAtelemetry = 45
lenNOAAchannel = 1040
lenNOAAline = 2080
lenNOAAframe = 64*lenNOAAline
fNOAACarrierHz = 2400.0

def _lineTelemetry(data):
    """ return mean and sdev of telemetry portions of single line data
    
    if len(data)==lenNOAAline (2080), return (Amean,Asdev,Bmean,Bsdev)

    if len(data)==lenNOAAchannel (1040), return (mean, sdev)

    """
    if len(data)==lenNOAAline:
        Adata = data[(lenNOAAchannel-lenNOAAtelemetry):lenNOAAchannel]
        Bdata =  data[(lenNOAAline-lenNOAAtelemetry):lenNOAAline]
        return (np.mean(Adata),np.std(Adata),np.mean(Bdata),np.std(Bdata))
    if len(data)==lenNOAAchannel:
        Tdata = data[(lenNOAAchannel-lenNOAAtelemetry):lenNOAAchannel]
        return (np.mean(Tdata),np.std(Tdata))
    raise Exception("cqwx.APT._lineTelemetry(data): invalid data length "+\
                    str(len(data)))

def _pulseSSR(data, pulse):
    """ return sum of square residuals of data vs syncA and space data
        
    """
    dataFromSync = data[0:lenNOAAsyncA]
    dataFromSpace = data[lenNOAAsyncA:(lenNOAAsyncA+lenNOAAspace)]
    ssq = np.sum(np.square(dataFromSync-pulse))+\
              lenNOAAspace*np.var(dataFromSpace)
    return ssq

def _G5(x):
    """return copy of x as (len(x)/5) x 5 array"""
    l = len(x)
    if l%5!=0:
        raise Exception("cqwx.APT._G5() input array size not divisible by 5")
    return np.copy(x).reshape(l/5,5)

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
        # 5 samples/byte*2080 bytes/line*2 APTlines/sec = 20800
        self.omega = 2*math.pi*fNOAACarrierHz/self.rate
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
        data = np.round(255*(demodSig-low)/delta)
        data[data<0]=0
        data[data>255]=255
        return data.astype(np.uint8)

    def _demodAM_by_hilbert(self, start=0, end=None):
        # see http://dsp.stackexchange.com/a/18800/11065
        if end is None:
            end = len(self.signal)
        end = min(end, len(self.signal))
        hilbert = scipy.signal.hilbert(self.signal[start:end])
        filtered = scipy.signal.medfilt(np.abs(hilbert), 5)
        return _G5(filtered)[:,2]

    def _data_from_hilbert(self):
        return self._digitize(self._demodAM_by_hilbert(), plow=1.0, phigh=99.0)

    def _demodAMrms(self, start=None, end=None, phase=None, denoise=False, domega=0.0):
        """ return root-mean-square demodulated signal

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
        of summing squared noise, and also creating a suprious noise signal at 
        twice omega, but that is summing over roughly an entire wave, and an 
        entire wave might sum the spurious signal "near" zero.
        
        """
        denoiseAmt = 0.0
        if start is None:
            start = 0
        if end is None:
            end = len(self.signal)
        ssq = np.sum(_G5(np.square(self.signal[start:end])), axis=1)
        if denoise is True:
            denoiseAmt = np.min(ssq)
            ssq -= denoiseAmt
        if phase is None:
            demod = np.sqrt(0.4*ssq)
        else:
            sqcos = np.square(np.cos(\
                (start+np.arange(end-start))*(self.omega+domega)+phase\
                                 ))
            divisor = np.sum(_G5(sqcos), axis=1)
            demod =  np.sqrt(np.divide(ssq, divisor))
        return demod

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

        self.rough_data = self._data_from_hilbert()

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

    def _ssr(self, start, phase=None, denoise=True, pulse=None, domega=0.0):
        length = 5*(lenNOAAsyncA+lenNOAAspace)
        raw = self._demodAM(start,\
                            start+length,\
                            phase,\
                            denoise,\
                            domega)
        data = self._digitize(raw)
        return _pulseSSR(data, pulse)

    def _findPhase(self, start, pulse=lNOAAsyncA):
        """return estimate of phase
        
        start: index to start of NOAA data line in self.signal
        end: index to end of NOAA data line in self.signal
 
        For proper operation, start must point to the signal index at the
        beginning of a line of data, i.e. at the beginning of a syncA pulse.  
       
        Determine the phase of the NOAA APT 2400 hz carrier tone using the
        portions of tbe signal called SYNC A and SPACE A. 

        SYNC A is a specific repeating high-low pulse
 
        SPACE A modulates the carrier
        at a constant level corresponding to white or black.  
       
        """
        length = 5*(lenNOAAsyncA+lenNOAAspace)
        if (len(self.signal)-start) < length:
            raise(Exception("findPhase: signal is of insufficient length"))
        M = [ self._ssr(start, phase, pulse=pulse) for phase in \
              -1.57+0.01*np.arange(314) ]
        phase = -1.57+0.01*np.argmin(M)
        return phase

    def _findDOmega(self, start, phase, jitter=0):
        domegas = (-1.57+0.01*np.arange(314))/(5*lenNOAAchannel)
        M = [ self._ssr(start+lenNOAAchannel+jitter,\
                        phase,\
                        pulse=lNOAAsyncB,\
                        domega=d) \
              for d in domegas]
        domega = domegas[np.argmin(M)]
        return domega

    def _findJitterPhase(self, start, pulse=lNOAAsyncA):
        j0 = 0
        j1 = 0
        end = start+5*lenNOAAsyncA+5*lenNOAAspace
        if start>2:
            j0=-2
        if (end+2)<len(self.signal):
            j1=2
        J = j0 + np.arange(1+j1-j0)
        Phases = [ self._findPhase(start+j) for j in J]
        ssq = [ self._ssr(start+j, Phases[i], pulse=pulse) \
                for (i,j) in enumerate(J) ]
        argminssq = np.argmin(ssq)
        return (J[argminssq], Phases[argminssq])

    def _residual(self, start, end, phase, demod):
        """ determine residual signal from demod amplitudes and phase
        
        input signals are self.signal[start:end] and
        a signal synthesiszed from demod, phase
        """
        l = end-start
        local_osc = np.cos((start+np.arange(l))*self.omega+phase)
        synth = local_osc*np.repeat(demod,5)
        return self.signal[start:end]-synth
        
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

    def printTelemetry(self):
        data = getattr(self, 'fine_data', self.rough_data)
        for line in data:
            print ",".join(map(str,_lineTelemetry(line)))
        

    def makePNG(self, fname, datasource):
        """create PNG file from decoded satellite image data
        
        fname:  file name to use in creating the png file.
        datasource:  either a STRING or a numpy array containing the image
        datasource:  'rough' -- use self.rough_data for image
                     'fine'  -- use self.fine_data for image
        
        
        """
        data = None
        if isinstance(datasource, str):
            if datasource == 'rough':
                data = np.copy(self.rough_data)
            if datasource == 'fine':
                data = np.copy(self.fine_data)
        if isinstance(datasource, np.ndarray):
            data = datasource
        if data is None:
            raise Exception("Fatal: makePNG requires datasource='fine' or 'rough' or numpy array")
        im = PIL.Image.fromarray(data)
        im.save(fname,'PNG')
        


def _NOAA_test_signal(minmod=0.05,maxmod=0.95,phase=0,domega=0,e=0,data=None):
    """return simulated NOAA signal with syncA pulse, space, and random data
    
    return length is 10400 byytes, equivalent to one NOAA line
    
    does not simulate syncB, spaceB, or telemetry bars.
    
    minmod: (0.05) minimum modulation level of 2400 hz carrier
    maxmod: (0.95) maximum modulation level of 2400 hz carrier
    phase: (0.0) phase of 2400 hz carrier at beginning of line
    domega: (0.0) difference in omega representing off-frequency carrier
    e: (0.0) standard deviation (signma) of Gaussian noise
    data: (if data is None) default to generating random data
          data may be specified for the non-syncA, non-space content
    
    """
    rate = 10.0*lenNOAAline
    omega = 2*math.pi*fNOAACarrierHz/rate
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
    signal = modulation*np.cos((omega+domega)*np.arange(5*lenNOAAline)+phase)+\
        e*np.random.normal(0.0,1.0,5*lenNOAAline)
    return (signal, linedata)


if __name__== "__main__":
    if len(sys.argv)<3:
        print __doc__
    rx = RX(sys.argv[1])
    if len(sys.argv)>=3:
        rx.rough_decode(pngfile=sys.argv[2])





