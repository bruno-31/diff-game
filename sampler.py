"""Sample random numbers according to a uniformly sampled 2d probability
density function.
The method used here probably is similar to rejection sampling
(http://en.wikipedia.org/wiki/Rejection_sampling).
"""

import numpy as np
import sys
from scipy.interpolate import RectBivariateSpline
import matplotlib.pylab as plt


class sampler(object):
    """Sampler object. To be instantiated with an (x,y) grid and a PDF
    function z = z(x,y).
    """

    def __init__(self, x, y, z, m=0.95, cond=None):
        """Create a sampler object from data.
        Parameters
        ----------
        x,y : arrays
            1d arrays for x and y data.
        z : array
            PDF of shape [len(x), len(y)]. Does not need to be normalized
            correctly.
        m : float, optional
            Number in [0; 1). Used as new maximum value in renormalization
            of the PDF. Random samples (x,y) will be accepted if
            PDF_renormalized(x, y) >= Random[0; 1). Low m values will create
            more values regions of low PDF.
        cond : function, optional
            A boolean function of x and y. True if the value in the x,y plane
            is of interest.
        Notes
        -----
        To restrict x and y to the unit circle, use
        cond=lambda x,y: x**2 + y**2 <= 1.
        For more information on the format of x, y, z see the docstring of
        scipy.interpolate.interp2d().
        Note that interpolation can be very, very slow for larger z matrices
        (say > 100x100).
        """

        # check validity of input:
        if(np.any(z < 0.0)):
            print >> sys.stderr("z has negative values and thus is not a density!")
            return

        if(not 0.0 < m < 1.0):
            print >> sys.stderr("m has to be in (0; 1)!")
            return

        maxVal = np.max(z)
        z *= m/maxVal  # normalize maximum value in z to m

        print("Preparing interpolating function")
        self._interp = RectBivariateSpline(x, y, z.transpose())  # TODO FIXME: why .transpose()?
        print("Interpolation done")
        self._xRange = (x[0], x[-1])  # set x and y ranges
        self._yRange = (y[0], y[-1])

        self._cond = cond

    def sample(self, size=1):
        """Sample a given number of random numbers with following given PDF.
        Parameters
        ----------
        size : int
            Create this many random variates.
        Returns
        -------
        vals : list
            List of tuples (x_i, y_i) of samples.
        """

        vals = []

        while(len(vals) < size):

            # first create x and y samples in the allowed ranges (shift from [0, 1)
            # to [min, max))
            while(True):
                x, y = np.random.rand(2)
                x = (self._xRange[1]-self._xRange[0])*x + self._xRange[0]
                y = (self._yRange[1]-self._yRange[0])*y + self._yRange[0]

                # additional condition true? --> use these values
                if(self._cond is not None):
                    if(self._cond(x, y)):
                        break
                    else:
                        continue
                else:  # no condition -> use values immediately
                    break

            # to decide if the values are to be kept, sample the PDF there and
            # decide about rejection
            chance = np.random.ranf()
            PDFsample = self._interp(x, y)

            # keep or reject sample? if at (x,y) the renormalized PDF is >= than
            # the random number generated, keep the sample
            if(PDFsample >= chance):
                vals.append((x, y))

        return vals

if(__name__ == '__main__'):  # test with an illustrative plot

    # create a sin^2*Gaussian PDF on the unit square and create random variates
    # inside the upper half of a disk centered on the middle of the square

    import matplotlib.pyplot as plt
    gridSamples = 1024
    x = np.linspace(0, 1., gridSamples)
    y = np.linspace(0, 1., gridSamples)
    XX, YY = np.meshgrid(x, y)
    # sample a sin^2*cos^2*Gaussian PDF (not normalized)
    z = np.exp(-(XX-0.5)**2/(2*0.2**2) - (YY-0.5)**2/(2*0.1**2))*np.sin(2*np.pi*XX)**2*np.cos(4*np.pi*(YY+XX))**2

    s = sampler(x, y, z, cond=lambda x, y: (x-0.5)**2 + (y-0.5)**2 <= 0.4**2 and y > 0.5)

    vals = s.sample(5000); xVals = []; yVals = []

    # plot sampled random variates over PDF
    plt.imshow(z, cmap=plt.cm.Blues, origin="lower",
               extent=(s._xRange[0], s._xRange[1], s._yRange[0], s._yRange[1]),
               aspect="equal")
    for item in vals:  # plot point by point
        xVals.append(item[0])
        yVals.append(item[1])
        plt.scatter(item[0], item[1], marker="x", c="red")
    plt.show()

    # create a histogram/density plot for random variates and plot over PDF
    hist, bla, blubb = np.histogram2d(xVals, yVals, bins=100, normed=True, range=((s._xRange[0], s._xRange[1]), (s._yRange[0], s._yRange[1])))
    plt.imshow(z, cmap=plt.cm.Blues, extent=(s._xRange[0], s._xRange[1], s._yRange[0], s._yRange[1]), aspect="equal", origin="lower")
    plt.title("'PDF'")
    plt.show()
    # have to plot transpose of hist because of NumPy's convention for histogram2d
    plt.imshow(hist.transpose(), cmap=plt.cm.Reds, extent=(s._xRange[0], s._xRange[1], s._yRange[0], s._yRange[1]), aspect="equal", origin="lower")
    plt.title("Density of random variates")
plt.show()