from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle


class HandlerColormap(HandlerBase):
    def __init__(self, cmap, fc, num_stripes=5, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
        self.fc = fc

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent],
                          width / self.num_stripes,
                          height,
                          fc=self.fc[i],
                          transform=trans)
            stripes.append(s)
        return stripes


def cm_to_inch(value):
    return value / 2.54