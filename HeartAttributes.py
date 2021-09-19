import numpy as np
class HeartAttributes():
    """
    Retains sliding window attributes for heart rate detection. Represents the attributes requires to calculate the heart
    rate for one person in sliding-window format.
    """

    WINDOW_LEN = 30 #moving window size
    THRESHOLD_TIME = 10 #if no data for this long, reset the tracker
    FPS = 24 #frame per second to process
    FT_WINDOW_LEN = 30
    HZ = FT_WINDOW_LEN

    def __init__(self):
        self.is_active = False
        self.reset()
        self.last_frame = 0

    def reset(self):
        """Resets states"""
        self.n_missing = 0 #number of missing frames with no data
        self.rgb_history = []
        self.rgb_ma = np.zeros((3,)) #moving average
        self.rgb_msa = np.zeros((3,))
        self.mas = []
        self.msas = []
        self.ft = np.zeros((self.FT_WINDOW_LEN * self.FPS))
        self.sig_history = []

    def update(self, rgb, frame):
        """Update state for one rgb observation, returns the signal"""
        if (frame - self.last_frame) > (self.THRESHOLD_TIME * self.FPS):
            self.reset()
        self.last_frame = frame

        self.rgb_history.append(rgb)

        if len(self.rgb_history) <= self.WINDOW_LEN * self.FPS:
            self.rgb_ma += (rgb - self.rgb_ma) / len(self.rgb_history)
            self.rgb_msa += (rgb ** 2 - self.rgb_msa) / len(self.rgb_history)
        else:
<<<<<<< HEAD
            self.rgb_ma += (rgb - self.rgb_history[-self.WINDOW_LEN * self.FPS]) / self.rgb_history

        return self.get_signal(rgb)
=======
            self.rgb_ma += (rgb - self.rgb_history[-(self.WINDOW_LEN * self.FPS)-1]) / (self.WINDOW_LEN * self.FPS)
            self.rgb_msa += (rgb ** 2 - self.rgb_history[-(self.WINDOW_LEN * self.FPS)-1] ** 2) / (self.WINDOW_LEN * self.FPS)

        self.mas.append(self.rgb_ma.copy())
        self.msas.append(self.rgb_msa.copy())
        
        sig = self.get_signal(rgb)

        self.sig_history.append(sig)

        #bounding window fourier transform
        N = self.FT_WINDOW_LEN * self.FPS
        idx = np.arange(N)
        if len(self.sig_history) <= N:
            sig_prevcycle = 0
        else:
            sig_prevcycle = self.sig_history[-N-1]
        self.ft = (self.ft + sig - sig_prevcycle) * (np.cos(idx*2*np.pi/N) + 1j*np.sin(idx*2*np.pi/N))

        return self.get_fourier()

    def is_acceptable(self, rgb):
        return (len(self.rgb_history) < 300) or not ((rgb - self.rgb_ma)**2 > 6*(self.rgb_msa - self.rgb_ma**2)).any() #5% statistical test
>>>>>>> b491f7759a604bb5e6344d5f7521e9a26dd953d6

    def get_signal(self, rgb):
        """
        Use Skin Tone Normalization to process RGB array signal in order to control
        for illumination, skin tone and spectral reflection

        rgb: (3, ) array of rgb values
        """

        # normalize for illumination by converting signal into what it would look like in white light
        rgb_st = np.array([0.7682, 0.5121, 0.3841]) #average skin tone
<<<<<<< HEAD
        rgb_w = rgb * rgb_st[None, :] / self.rgb_ma

=======
        rgb_w = rgb * rgb_st / self.rgb_ma
        
>>>>>>> b491f7759a604bb5e6344d5f7521e9a26dd953d6
        # calculate chrominance signals, which removes noise due to specular reflection
        X = (rgb_w[0] - rgb_w[1]) / (rgb_st[0] - rgb_st[1])
        Y = (rgb_w[0] + rgb_w[1] - 2*rgb_w[2]) / (rgb_st[0] + rgb_st[1] - 2*rgb_st[2])

        # calculate signal X/Y which cancels out brightness effects
        return X/Y
<<<<<<< HEAD
=======

    def get_fourier(self):
        return np.absolute(self.ft)
>>>>>>> b491f7759a604bb5e6344d5f7521e9a26dd953d6
