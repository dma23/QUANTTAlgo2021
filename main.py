import numpy as np
import pandas as pd


"""
A library to simulate Geometric fractal brownian motion (not supported by QuantConnnect)
Our code begins on line 286
"""
class FBM(object):
    """The FBM class.
    After instantiating with n = number of increments, hurst parameter, length
    of realization (default = 1) and method of generation
    (default daviesharte), call fbm() for fBm, fgn()
    for fGn, or times() to get corresponding time values.
    """

    def __init__(brownian, n, hurst, length=1, method="daviesharte"):
        """Instantiate the FBM."""
        brownian._methods = {"daviesharte": brownian._daviesharte, "cholesky": brownian._cholesky, "hosking": brownian._hosking}
        brownian.n = n
        brownian.hurst = hurst
        brownian.length = length
        brownian.method = method
        brownian._fgn = brownian._methods[brownian.method]
        # Some reusable values to speed up Monte Carlo.
        brownian._cov = None
        brownian._eigenvals = None
        brownian._C = None
        # Flag if some params get changed
        brownian._changed = False

    def __str__(brownian):
        """Str method."""
        return (
            "fBm ("
            + str(brownian.method)
            + ") on [0, "
            + str(brownian.length)
            + "] with Hurst value "
            + str(brownian.hurst)
            + " and "
            + str(brownian.n)
            + " increments"
        )

    def __repr__(brownian):
        """Repr method."""
        return (
            "FBM(n="
            + str(brownian.n)
            + ", hurst="
            + str(brownian.hurst)
            + ", length="
            + str(brownian.length)
            + ', method="'
            + str(brownian.method)
            + '")'
        )

    @property
    def n(brownian):
        """Get the number of increments."""
        return brownian._n

    @n.setter
    def n(brownian, value):
        if not isinstance(value, int) or value <= 0:
            raise TypeError("Number of increments must be a positive int.")
        brownian._n = value
        brownian._changed = True

    @property
    def hurst(brownian):
        """Hurst parameter."""
        return brownian._hurst

    @hurst.setter
    def hurst(brownian, value):
        if not isinstance(value, float) or value <= 0 or value >= 1:
            raise ValueError("Hurst parameter must be in interval (0, 1).")
        brownian._hurst = value
        brownian._changed = True

    @property
    def length(brownian):
        """Get the length of process."""
        return brownian._length

    @length.setter
    def length(brownian, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Length of fbm must be greater than 0.")
        brownian._length = value
        brownian._changed = True

    @property
    def method(brownian):
        """Get the algorithm used to generate."""
        return brownian._method

    @method.setter
    def method(brownian, value):
        if value not in brownian._methods:
            raise ValueError("Method must be 'daviesharte', 'hosking' or 'cholesky'.")
        brownian._method = value
        brownian._fgn = brownian._methods[brownian.method]
        brownian._changed = True

    def fbm(brownian):
        """Sample the fractional Brownian motion."""
        return np.insert(brownian.fgn().cumsum(), [0], 0)

    def fgn(brownian):
        """Sample the fractional Gaussian noise."""
        scale = (1.0 * brownian.length / brownian.n) ** brownian.hurst
        gn = np.random.normal(0.0, 1.0, brownian.n)

        # If hurst == 1/2 then just return Gaussian noise
        if brownian.hurst == 0.5:
            return gn * scale
        else:
            fgn = brownian._fgn(gn)

        # Scale to interval [0, L]
        return fgn * scale

    def times(brownian):
        """Get times associated with the fbm/fgn samples."""
        return np.linspace(0, brownian.length, brownian.n + 1)

    def _autocovariance(brownian, k):
        """Autocovariance for fgn."""
        return 0.5 * (abs(k - 1) ** (2 * brownian.hurst) - 2 * abs(k) ** (2 * brownian.hurst) + abs(k + 1) ** (2 * brownian.hurst))

    def _daviesharte(brownian, gn):
        """Generate a fgn realization using Davies-Harte method.
        Uses Davies and Harte method (exact method) from:
        Davies, Robert B., and D. S. Harte. "Tests for Hurst effect."
        Biometrika 74, no. 1 (1987): 95-101.
        Can fail if n is small and hurst close to 1. Falls back to Hosking
        method in that case. See:
        Wood, Andrew TA, and Grace Chan. "Simulation of stationary Gaussian
        processes in [0, 1] d." Journal of computational and graphical
        statistics 3, no. 4 (1994): 409-432.
        """
        # Monte carlo consideration
        if brownian._eigenvals is None or brownian._changed:
            # Generate the first row of the circulant matrix
            row_component = [brownian._autocovariance(i) for i in range(1, brownian.n)]
            reverse_component = list(reversed(row_component))
            row = [brownian._autocovariance(0)] + row_component + [0] + reverse_component

            # Get the eigenvalues of the circulant matrix
            # Discard the imaginary part (should all be zero in theory so
            # imaginary part will be very small)
            brownian._eigenvals = np.fft.fft(row).real
            brownian._changed = False

        # If any of the eigenvalues are negative, then the circulant matrix
        # is not positive definite, meaning we cannot use this method. This
        # occurs for situations where n is low and H is close to 1.
        # Fall back to using the Hosking method. See the following for a more
        # detailed explanation:
        #
        # Wood, Andrew TA, and Grace Chan. "Simulation of stationary Gaussian
        #     processes in [0, 1] d." Journal of computational and graphical
        #     statistics 3, no. 4 (1994): 409-432.
        if np.any([ev < 0 for ev in brownian._eigenvals]):
            warnings.warn(
                "Combination of increments n and Hurst value H "
                "invalid for Davies-Harte method. Reverting to Hosking method."
                " Occurs when n is small and Hurst is close to 1. "
            )
            # Set method to hosking for future samples.
            brownian.method = "hosking"
            # Don"t need to store eigenvals anymore.
            brownian._eigenvals = None
            return brownian._hosking(gn)

        # Generate second sequence of i.i.d. standard normals
        gn2 = np.random.normal(0.0, 1.0, brownian.n)

        # Resulting sequence from matrix multiplication of positive definite
        # sqrt(C) matrix with fgn sample can be simulated in this way.
        w = np.zeros(2 * brownian.n, dtype=complex)
        for i in range(2 * brownian.n):
            if i == 0:
                w[i] = np.sqrt(brownian._eigenvals[i] / (2 * brownian.n)) * gn[i]
            elif i < brownian.n:
                w[i] = np.sqrt(brownian._eigenvals[i] / (4 * brownian.n)) * (gn[i] + 1j * gn2[i])
            elif i == brownian.n:
                w[i] = np.sqrt(brownian._eigenvals[i] / (2 * brownian.n)) * gn2[0]
            else:
                w[i] = np.sqrt(brownian._eigenvals[i] / (4 * brownian.n)) * (gn[2 * brownian.n - i] - 1j * gn2[2 * brownian.n - i])

        # Resulting z is fft of sequence w. Discard small imaginary part (z
        # should be real in theory).
        z = np.fft.fft(w)
        fgn = z[: brownian.n].real
        return fgn

    def _cholesky(brownian, gn):
        """Generate a fgn realization using the Cholesky method.
        Uses Cholesky decomposition method (exact method) from:
        Asmussen, S. (1998). Stochastic simulation with a view towards
        stochastic processes. University of Aarhus. Centre for Mathematical
        Physics and Stochastics (MaPhySto)[MPS].
        """
        # Monte carlo consideration
        if brownian._C is None or brownian._changed:
            # Generate covariance matrix
            G = np.zeros([brownian.n, brownian.n])
            for i in range(brownian.n):
                for j in range(i + 1):
                    G[i, j] = brownian._autocovariance(i - j)

            # Cholesky decomposition
            brownian._C = np.linalg.cholesky(G)
            brownian._changed = False

        # Generate fgn
        fgn = np.dot(brownian._C, np.array(gn).transpose())
        fgn = np.squeeze(fgn)
        return fgn

    def _hosking(brownian, gn):
        """Generate a fGn realization using Hosking's method.
        Method of generation is Hosking's method (exact method) from his paper:
        Hosking, J. R. (1984). Modeling persistence in hydrological time series
        using fractional differencing. Water resources research, 20(12),
        1898-1908.
        """
        fgn = np.zeros(brownian.n)
        phi = np.zeros(brownian.n)
        psi = np.zeros(brownian.n)
        # Monte carlo consideration
        if brownian._cov is None or brownian._changed:
            brownian._cov = np.array([brownian._autocovariance(i) for i in range(brownian.n)])
            brownian._changed = False

        # First increment from stationary distribution
        fgn[0] = gn[0]
        v = 1
        phi[0] = 0

        # Generate fgn realization with n increments of size 1
        for i in range(1, brownian.n):
            phi[i - 1] = brownian._cov[i]
            for j in range(i - 1):
                psi[j] = phi[j]
                phi[i - 1] -= psi[j] * brownian._cov[i - j - 1]
            phi[i - 1] /= v
            for j in range(i - 1):
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]
            v *= 1 - phi[i - 1] * phi[i - 1]
            for j in range(i):
                fgn[i] += phi[j] * fgn[i - j - 1]
            fgn[i] += np.sqrt(v) * gn[i]

        return fgn


def fbm(n, hurst, length=1, method="daviesharte"):
    """One off sample of fBm."""
    f = FBM(n, hurst, length, method)
    return f.fbm()


def fgn(n, hurst, length=1, method="daviesharte"):
    """One off sample of fGn."""
    f = FBM(n, hurst, length, method)
    return f.fgn()


def times(n, length=1):
    """Generate the times associated with increments n and length."""
    return np.linspace(0, length, n + 1)








class CreativeFluorescentOrangeButterfly(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)  # Set Start Date
        self.SetCash(100000)  # Set Strategy Cash
        
        self.stock = []
        
        stocks = [
            "AMD",
            "AAPL",
            "MSFT",
            "TSLA"]
            
        for tick in stocks: 
            self.stock.append(self.AddEquity(tick, Resolution.Daily))

        #self.tsla = self.AddEquity("TSLA", Resolution.Daily)
        #self.aapl = self.AddEquity("AAPL", Resolution.Daily)


    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        
        """
        symbToTimeSeries
            A function that takes in data from QuantConnect and converts it to an np.ndarray

            Params:
            self: an object used by quantconnect
            symbol: a symbol object from quantconnect
            
        """
        
        def symbToTimeSeries(self, symbol):
            history = self.History(symbol, 90, Resolution.Daily)
            timeSeries = history[['close']].to_numpy
            return timeSeries
        

        """
        Notes for whoever is using this, under the definition of Bollinger Bands on TradingView what this program should be
        doing is creating a band of three lines which are plotted in relation to security prices. The middle line is usually the
        SMA (simple moving average) but we're using a different method here in the form of GFBM.

        - Standard Brownian Motion is a stochastic process which is essentially a collection of random variables defined on a
          common probability space

        - This specific stochastic process models a random walk process which means it will randomly move either up or down
          simulating somewhat how the stock market will move up and down

        - STANDARD Brownian Motion ==> B(t) ~ N(0, t)

        - X(t) is a Brownian Motion PROCESS if
                                X(t) = ( volatility * B(t) ) + ( drift * t )
                                                               |  ^mean^  |

         where {B(t), t>=0} is a STANDARD Brownian Motion

        - Geometric Brownian Motion can be defined as:
                                            Y(t) = exp(X(t))
          where X(t) is a Brownian Motion PROCESS

        Now the issue with using regular GBM is that it assumes everything is random with no correlation but the stock market
        does not necessarily always act in such a manner. It will sometimes follow trends and sometimes will revert to mean.
        This is where Geometric FRACTIONAL Brownian Motion comes in. This is a generalization of GBM and uses the Hurst exponent
        to also model mean-reverting and trending processes along with random walk.
        """

        # Function that returns the hurst exponent taking in a time series of past stock data, along with a maximum lag that
        # the user wants to use to calculate.

        def hurst(time_series, max_lag=20):
            # time_series = price data on specific stock we want to calculate the hurst exponent for
            # max_lag = this is the maximum lag we will calculate up to


            """Returns the Hurst Exponent of the time series"""

            # creates an array of lag values starting from the minimum lag of 2, and goes up to the max_lag which defaults to 20
            lags = range(2, max_lag)
            # finds variances of the lagged differences
            tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
            # calculate the slope of the log plot this returns the Hurst Exponent
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]

        """
        The Hurst exponent will indicate for a specific stock whether it is random walk (h = 0.5), mean reverting (h < 0.5), or
        trending (h > 0.5). With this we will be able to predict more accurate future prices than if we were to use regular GBM.

        Now we will plug the hurst exponent into our GFBM equation (which was derived in the paper sent by Gabe) to create a
        series of predicted values of which we will then use as the median of our risk ranges. We will first need to calculate
        the drift and volatility based on past stock data
        """

        """
        This is one of two functions that will be used as precursors to using our main function, the GFBM. This function will
        find the drift of our stock data, the change of the average price of the stock.
        """


        def calcDrift(n, time_series):
            """
            :param n: the number of elements in the series
            :param time_series: the series of stock price data we will be iterating over
            :return: the change of the average value of a stochastic process (in our case the average stock price)
            """


            sum = 0
            for i in range(1, n - 1):
                sum += np.log(time_series[i] / time_series[i - 1])
            drift = sum / n
            return drift


        """
        This is the second of the two functions, this must be executed after the drift function as it requires the drift in
        order to work. This function finds the volatility of the stock data.
        """""




        def calcVolatility(n, time_series, drift, time_step=1):
            """
            :param n: the number of elements in the series
            :param time_series: the series of stock price data we will be iterating over
            :param drift: the change of the average value of a stochastic process (in our case the average stock price)
            :param time_step:  this timeStep variable will have to be an approximation based on what the average time
                                difference is between stocks, but in the case where we decide to only use closing values this
                                can remain as being equal to one
            :return: returns the sample stochastic volatility of the stock data
            """
            sum = 0
            for i in range(1, n - 1):
                sum += (np.log(time_series[i] / time_series[i - 1]) - drift) ** 2
            variance = sum / (n - 1)
            stdev = np.sqrt(variance)
            vol = stdev / np.sqrt(time_step)
            return vol
        """
        Using these we finally have all the required data in order to calculate the GFBM
        """



        # returns a time series of a simulated Geometric Fractional Brownian Motion
        def GFBM(initS, drift, volatility, hurst, n, length=15):
            """
            :param initS: the initial stock price at the time we calculate the GFBM
            :param drift: the change of the average value of a stochastic process (in our case the average stock price)
            :param volatility: the volatility of the stock data based on past data
            :param hurst: the hurst exponent calculated in the hurst function
            :param n: the number of equispaced increments desired for a fBm with hurst parameter hurst
            :param length: the size of the interval that is being broken up into the n equispaced increments
            :return: a generated GFBM
            """

            # this just generates a FBM process using the daviesharte method (which is the most efficient way of calculating it
            # as it is O(nlogn) and the other two are O(n^2)
            fbm_series = fbm(length, hurst, length, "daviesharte")
            # I don't really know how to declare an np.array (and if it's even better to use here) but feel free to change this
            # to use an np.array instead
            gfbm = []
            # Now create the time series by generating the GFBM equation
            # What we do here now is if it is the first timestep, we multiply the GFBM by the initial stock price
            # Otherwise we use the previously generated stock price to make our next decision, this follows from
            # the idea that this stochastic process has no long-term memory
            for i in range(1, length):
                if i == 1:
                    gfbm.append(initS * np.exp((drift - (volatility ** 2) / 2) * i + volatility * fbm_series[i - 1]))
                else:
                    gfbm.append(gfbm[i - 2] * np.exp((drift - (volatility ** 2) / 2) * i + volatility * fbm_series[i - 1]))
            return np.asarray(gfbm)


        """
        NOW WE BEGIN IMPLEMENTATION
        """
        
        for name in self.stock:
            symbToTimeSeries(self, name.Symbol)
        

        length = 90  # TRADE length
        trendLength = 90  # Trend Length, this defaults to 63 periods, 3 months or more...
        resolution = 0  # Resolution (the size of one period, hourly daily, weekly etc.)


        """
        I have no idea what these mean but these were in the code Gabe sent over
        Also now that I think about it, resolution can easily be interpreted as being n in the previous functions
        """


        timeframe = self.History(name.Symbol, 90, Resolution.Daily)
        securityclose = timeframe[['close']].to_numpy()
        securityhigh = timeframe[['high']].to_numpy()
        securitylow = timeframe[['low']].to_numpy()

        """
        This variable is created just for ease of implementation as arrays start from index 0 not 1
        """

        lengthMinus1 = length - 1

        # calculates the max difference between a time series and a line defined by a slope (s) intersecting the data point at
        # src[n] over a number (n) of periods

        def max_diff(src, s, n):
            m = -100000000.0
            for i in range(0, n):
                m = max(m, src[n - i] - (src[n] + (s * i)))
            return m


        # calculates the min difference between a time series and a line defined by a slope (s) intersecting the data point at
        # src[n] over a number (n) of periods

        def min_diff(src, s, n):
            m = 100000000.0
            for i in range(0, n):
                m = min(m, src[n - i] - (src[n] + (s * i)))
            return m
        """
        Compute a slope series where the value at every period is the slope of a line connecting the closing price at
        that period with the close at the period TRADE length prior.  This is the dashed line in the diagram on page
        179 of Mandelbrot.
        """

        slope = (securityclose[0] - securityclose[lengthMinus1]) / lengthMinus1
        """
        Compute a Minimum Difference series where each point is the min value in the diagram on page 179 of Mandelbrot over
        a TRADE length.
        """

        mindiff = min_diff(securityclose, slope[0], lengthMinus1)

        """
        Compute the corresponding Maximum Difference series where each point is the max value in the diagram on page 179 of
        Mandelbrot over a TRADE length.
        """

        maxdiff = max_diff(securityclose, slope[0], lengthMinus1)

        """
        Create the Bridge Range series by adding these to the closing price series.  These bound R(t,d) in the diagram on page
        179 of Mandelbrot.
        """

        bridge_range_bottom = securityclose[:89] + mindiff
        bridge_range_top = securityclose[:89] + maxdiff

        """
        As of now I've had the Hurst exponent calculated using the closing prices of the securities but we could possibly have
        to use a different time series (I was thinking more real time data of the changes of the prices to match up with our
        strategy)
        """

        hurst = hurst(securityclose)

        """
        Create the Bollinger Bands.  Bollinger Bands are a channel created by two lines plotted a number of standard deviations
        off a moving average of the price.  Here, we are using a GFBM of the closing price for a TRADE duration, with the
        channels plotted 2 standard deviations (of the closing price) above and below that.
        """

        dr = calcDrift(length, securityclose)
        volatility = calcVolatility(length, securityclose, dr)
        sd = np.std(securityclose)

        # think of n as the number of trades we want to consider within the 15-day period we are dealing with. Feel free to
        # change n as needed (I've set it to 100 arbitrarily)

        gfbm = GFBM(securityclose[0], dr, volatility, hurst, 100, length)


        """
        Compute the Bridge Band by interpolation between Bridge Ranges and Bollinger Bands based upon the value of the Hurst
        exponent
        """

        bb_bottom = gfbm - (sd * 2)
        bb_top = gfbm + (sd * 2)


        bridge_band_bottom = bb_bottom + ((bridge_range_bottom - bb_bottom) * abs((hurst * 2) - 1))
        bridge_band_top = bb_top - ((bb_top - bridge_range_top) * abs((hurst * 2) - 1))

        """
        Create 3 additional series which divide the band into quartiles.  These are used in the suggested positioning
        calculations
        """

        bridge_band_mid = bridge_band_bottom + ((bridge_band_top - bridge_band_bottom) / 2)
        bridge_band_bottom_mid = bridge_band_bottom + ((bridge_band_mid - bridge_band_bottom) / 2)
        bridge_band_top_mid = bridge_band_mid + ((bridge_band_top - bridge_band_mid) / 2)


        def highest(src, length):
            high = src[-1]
            for i in range(2, length):
                high = max(high, src[-i])
            return high


        def lowest(src, length):
            low = src[-1]
            for i in range(2, length):
                low = min(low, src[-i])
            return low


        """
        The Trend Line is the middle channel of a Donchian Channel.  The upper channel is the highest price in the last N days -
        where N is the TREND length.  The lower channel is the lowest price in the last N days.  The middle channel is the
        average of the two - what we are calculating in the statement below.
        """
        trend = lowest(securitylow, trendLength) + ((highest(securityhigh, trendLength) - lowest(securitylow, trendLength)) / 2)


        """
        The Bullish/Bearish sentiment a simple comparison of the closing price to the trend line.  If the closing price
        is above the trend we are Bullish, if the closing price is below the trend we are Bearish.
        """

        if securityclose[0] > trend:
            self.SetHoldings(name.Symbol, 0.1)
        
        #elif securityclose[0] <= trend:
        #   self.SetHoldings(name.Symbol, -0.1)
        
        
