import numpy as np
class HeartAttributes():
    """
    Retains sliding window attributes for heart rate detection. Represents the attributes requires to calculate the heart 
    rate for one person in sliding-window format. 
    """

    WINDOW_LEN = 6 #moving window size
    THRESHOLD_TIME = 3 #if no data for this long, reset the tracker
    FPS = 10 #frame per second to process

    def __init__(self):
        self.is_active = False
        self.reset()
        self.last_frame = 0

    def reset(self):
        """Resets states"""
        self.n_missing = 0 #number of missing frames with no data
        self.rgb_history = []
        self.rgb_ma = np.zeros((3,)) #moving average

    def update(self, rgb, frame):
        """Update state for one rgb observation, returns the signal"""
        if (frame - self.last_frame) > (self.THRESHOLD_TIME * self.FPS):
            self.reset()
        self.last_frame = frame

        self.rgb_history.append(rgb)

        if len(self.rgb_history) <= self.WINDOW_LEN * self.FPS:
            self.rgb_ma += (rgb - self.rgb_ma) / self.rgb_history
        else:
            self.rgb_ma += (rgb - self.rgb_history[-self.WINDOW_LEN * self.FPS]) / self.rgb_history
        
        return self.get_signal(rgb)

    def get_signal(self, rgb):
        """
        Use Skin Tone Normalization to process RGB array signal in order to control
        for illumination, skin tone and spectral reflection

        rgb: (T, 3) array of rgb values, T >> 1
        """

        # normalize for illumination by converting signal into what it would look like in white light
        rgb_st = np.array([0.7682, 0.5121, 0.3841]) #average skin tone
        rgb_w = rgb * rgb_st[None, :] / self.rgb_ma
        
        # calculate chrominance signals, which removes noise due to specular reflection
        X = (rgb_w[0] - rgb_w[1]) / (rgb_st[0] - rgb_st[1])
        Y = (rgb_w[0] + rgb_w[1] - 2*rgb[2]) / (rgb_st[0] + rgb_st[1] - 2*rgb_st[2])

        # calculate signal X/Y which cancels out brightness effects
        return X/Y