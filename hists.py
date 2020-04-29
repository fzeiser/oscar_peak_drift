import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, Normalize
from ompy import Matrix
from scipy.signal import find_peaks
import numpy.ma as ma
from scipy.optimize import curve_fit
from typing import Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import re
from datetime import datetime

from scipy.linalg import lstsq


def read_data(fname, max_rows=32769):
    data = np.loadtxt(fname, skiprows=1, max_rows=max_rows)
    return data


def calibrate(coords, y=[1100, 1300]):
    dets = set(coords[:, 1])
    pars = np.zeros((len(dets), 2))
    for det in dets:
        xy = coords[coords[:, 1] == det]
        x = xy[:, 0]
        M = x[:, np.newaxis]**[0, 1]
        p, _, _, _ = lstsq(M, y)
        pars[det] = p
    return pars


def plot_as_heatmap(ax, x, y, det=None, vmin=None, vmax=None,
                    scale="log", **kwargs):
    if scale == 'log':
        if vmin is not None and vmin <= 0:
            raise ValueError("`vmin` must be positive for log-scale")
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif scale == 'linear':
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError("Unsupported zscale ", scale)

    extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2., det, det+.9]
    ax.imshow(y[np.newaxis, :], aspect="auto", extent=extent,
              norm=norm)
    ax.set_yticks([det])
    ax.set_xlim(extent[0], extent[1])


def save_coords_from_click(fig, fname="coords.txt"):
    try:
        coords = np.loadtxt(fname)
        print("loading file (instead)")
        # plt.show()
    except OSError:
        coords = []

        def onclick(event):
            """ depends on global `coords` """
            ix, iy = event.xdata, event.ydata
            print(f'x = {ix:.0f}, y = {iy:.0f}')
            coords.append((ix, iy))
            return coords

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)

        np.savetxt(fname, np.array(coords),
                   header="x y")
    coords = np.array(coords)
    coords = coords.astype(int)
    return coords


def avg_around_peak(x1, x2, x, y, width=30):
    xmask = ma.masked_outside(x, x1, x2)
    ymask = ma.masked_array(y, mask=xmask.mask)
    imax = ymask.argmax()

    xpeak = x[imax]
    xmask = ma.masked_outside(x, xpeak-width/2, xpeak+width/2)
    ymask = ma.masked_array(y, mask=xmask.mask)
    avg = np.average(xmask, weights=ymask)
    return avg


def FitPeak(*args, **kwargs) -> Dict[str, float]:
    """ wrapper for _FitPeak """
    try:
        return _FitPeak(*args, **kwargs)
    except RuntimeError:
        print("cannot estimate; return dummy")
        return {'const': np.nan, 'mean': np.nan, 'std': np.nan,
                'slope': np.nan, 'intercept': np.nan}


def _FitPeak(x, y, pre_region: Tuple[float, float],
             post_region: Tuple[float, float]) -> Dict[str, float]:
    """Fit a gamma peak with a gaussian on top of a linear
    spectra.

    Args:
        pre_region: Energy region with linear spectra on the left
        side of the peak
        post_region: Energy region with linear spectra on the right
        side of the peak
    """
    def ix(x0):
        return (np.abs(x-x0)).argmin()

    pre_slice = slice(ix(pre_region[0]), ix(pre_region[1])+1)
    peak_slice = slice(ix(pre_region[1])+1, ix(post_region[0]))
    post_slice = slice(ix(post_region[0]), ix(post_region[1])+1)
    fit_slice = slice(ix(pre_region[0]), ix(post_region[1])+1)

    pre_spec = y[pre_slice]
    peak_spec = y[peak_slice]
    post_spec = y[post_slice]

    fit_spec = y[fit_slice]

    # Estimate the mean, std and constant
    peak_mean = np.sum(x[peak_slice]*peak_spec)/np.sum(peak_spec)
    peak_var = np.sum(peak_spec*x[peak_slice]**2)/np.sum(peak_spec)
    peak_std = np.sqrt(peak_var - peak_mean**2)

    # We estimate the constant from the height found at the mean
    peak_const = np.max(peak_spec)/Gaus(peak_mean, 1., peak_mean, peak_std)

    # Estimate the linear background
    pol_estimate = np.polyfit(np.append(x[pre_slice], x[post_slice]),
                              np.append(pre_spec, post_spec), 1)

    # Calculate the curve fitting
    initial_guess = [peak_const, peak_mean, peak_std,
                     pol_estimate[0], pol_estimate[1]]
    popt, cov = curve_fit(Model, x[fit_slice], fit_spec, p0=initial_guess)
    return {'const': popt[0], 'mean': popt[1], 'std': popt[2],
            'slope': popt[3], 'intercept': popt[4]}


def Gaus(x, const, mean, std):
    return const*np.exp(-0.5*((x-mean)/std)**2)/(std*np.sqrt(2*np.pi))


def Pol(x, slope, intercept):
    return x*slope + intercept


def Model(x, const, mean, std, slope, intercept):
    return Gaus(x, const, mean, std) + Pol(x, slope, intercept)

def run_number_keys(text):
    '''
    find run number eg in hist/run82_mod0.asc -> 82
    '''
    return re.search(r'(\d+)(?!run)(?=_)', text).group(0)

def analysis_per_module(module="mod0"):
    """ All too long script for the whole analysis per module

    Idea:
    - For run0, do:
        1) Rough calibration of all detectors by hand using the 1.1 and 1.3 MeV
           peaks
        2) improve this calibrarion a bit
        3) better, "final" calibration by peak fitting to 120, 344 and 1330 keV
           peaks.
           Note that the 1178 keV peak doens't seem to work well; overlapping
           peaks?
    - Use this calibration for all runs, but to peakfitting for each run
      separately
    - Plot devation of peak location from mean for all runs
    """
    figdir = Path("figs")
    savedir = Path("save")
    for d in [figdir, savedir]:
        d.mkdir(exist_ok=True)

    path = Path("hist")
    files = [fn for fn in sorted(path.iterdir())
             if fn.match(f"{module}_*.asc")]
    times = [datetime.strptime(file.name, f"{module}_%Y%m%d-%H%M%S.asc")
             for file in files]

    print("start loading files")
    runs = [read_data(fn, max_rows=4000) for fn in files]
    print(f"Analysing for {len(runs)} runs")
    ndets = 15


    # plot initial & calibrate roughly
    fig, ax = plt.subplots()
    fig.suptitle("no calibration: raw")
    mat = Matrix(values=runs[0].T)
    mat.plot(ax=ax, scale="log")
    coords = save_coords_from_click(fig, fname=savedir/f"coords_{module}.txt")
    pars = calibrate(coords, y=[1100, 1300])
    fig.savefig(figdir / f"calib1_{module}.jpeg")


    # plot 1st calig & calibrate more
    fig, ax = plt.subplots(ndets, 1, sharex=True)
    fig.suptitle("after first calib")
    ax = ax[::-1]
    for i in np.arange(ndets):
        y = runs[0][:, i]
        x = np.arange(len(y))*pars[i, 1] + pars[i, 0]
        mat = Matrix(values=y, Eg=x, Ex=[1])
        plot_as_heatmap(ax=ax[i], x=x, y=y, scale="log", det=i)
    coords2 = save_coords_from_click(fig,
                                     fname=savedir/f"coords2_{module}.txt")
    pars2 = calibrate(coords2, y=[344, 1100, 1300])
    fig.savefig(figdir / f"calib2_{module}.jpeg")


    # plot 2st calig & ...
    fig, ax = plt.subplots(ndets, 1, sharex=True)
    fig.suptitle("after 2nd calib")
    ax = ax[::-1]
    for i in np.arange(ndets):
        y = runs[0][:, i]
        x = np.arange(len(y))*pars[i, 1] + pars[i, 0]
        x = x*pars2[i, 1] + pars2[i, 0]
        plot_as_heatmap(ax=ax[i], x=x, y=y, scale="log", det=i)
    fig.savefig(figdir / f"calib3_{module}.jpeg")

    # ... and make final calibration
    coords3 = []
    for i, det in enumerate(np.arange(ndets)):
        y = runs[0][:, i]
        x = np.arange(len(y))*pars[i, 1] + pars[i, 0]
        x = x*pars2[i, 1] + pars2[i, 0]

        # this doesn't work well due to background!
        # x0 = avg_around_peak(110, 130, x, y, width=15)
        # x1 = avg_around_peak(320, 360, x, y)
        # x2 = avg_around_peak(1050, 1200, x, y)
        # x3 = avg_around_peak(1250, 1380-70, x, y)  # calib slightly off

        px0 = FitPeak(x, y, pre_region=[100, 110], post_region=[140, 150])
        px1 = FitPeak(x, y, pre_region=[270, 300], post_region=[360, 380])
        # px2 = FitPeak(x, y, pre_region=[1080, 1100], post_region=[1150, 1180])
        px3 = FitPeak(x, y, pre_region=[1180, 1230], post_region=[1315, 1320])

        # fig2, ax2 = plt.subplots()
        # ax2.plot(x, y)
        # ax2.plot(x, Model(x, **px0))
        # ax2.plot(x, Model(x, **px1))
        # # ax2.plot(x, Model(x, **px2))
        # ax2.plot(x, Model(x, **px3))
        # ax2.set_yscale("log")
        # plt.show()

        ax[i].axvline(x=px0["mean"])
        ax[i].axvline(x=px1["mean"])
        # ax[i].axvline(x=px2["mean"])
        ax[i].axvline(x=px3["mean"])

        coords3.append([px0["mean"], det])
        coords3.append([px1["mean"], det])
        # coords3.append([px2["mean"], det])
        coords3.append([px3["mean"], det])
    coords3 = np.array(coords3, dtype=int)

    # something fishy with the 1173: Overlap of resonances?
    # pars3 = calibrate(coords3, y=[121, 344, 1173, 1333])
    pars3 = calibrate(coords3, y=[121, 344, 1333])

    # just a check of the final calibration
    fig, ax = plt.subplots(ndets, 1, sharex=True)
    fig.suptitle("after final calib")
    ax = ax[::-1]
    for i in np.arange(ndets):
        y = runs[0][:, i]
        x = np.arange(len(y))*pars[i, 1] + pars[i, 0]
        x = x*pars2[i, 1] + pars2[i, 0]
        x = x*pars3[i, 1] + pars3[i, 0]

        # print("det:", i)
        px0 = FitPeak(x, y, pre_region=[100, 110], post_region=[140, 150])
        px1 = FitPeak(x, y, pre_region=[270, 300], post_region=[360, 380])
        # px2 = FitPeak(x, y, pre_region=[1080, 1100], post_region=[1150, 1180])
        px3 = FitPeak(x, y, pre_region=[1230, 1290], post_region=[1365, 1370])

        ax[i].axvline(x=px0["mean"])
        ax[i].axvline(x=px1["mean"])
        # ax[i].axvline(x=px2["mean"])
        ax[i].axvline(x=px3["mean"])

        plot_as_heatmap(ax=ax[i], x=x, y=y, scale="log", det=i)

    # now find peaks for all runs and all detectors
    # using the calibration from the fist run & detector
    fname = savedir / f"means_{module}.npy"
    npeaks = 3
    try:
        means = np.load(fname)
        print(f"loaded means from {fname}")
    except OSError:
        means = np.zeros((len(runs), ndets, npeaks))
        for irun, arr in enumerate(tqdm(runs)):
            for det in range(ndets):
                y = arr[:, det]

                x = np.arange(len(y))*pars[det, 1] + pars[det, 0]
                x = x*pars2[det, 1] + pars2[det, 0]
                x = x*pars3[i, 1] + pars3[i, 0]

                px0 = FitPeak(x, y, pre_region=[100, 110],
                              post_region=[140, 150])
                px1 = FitPeak(x, y, pre_region=[270, 300],
                              post_region=[360, 380])
                px3 = FitPeak(x, y, pre_region=[1230, 1290],
                              post_region=[1365, 1370])

                means[irun, det, :] = [px0["mean"], px1["mean"], px3["mean"]]
        np.save(fname, means)

    # plot all together
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    fig.suptitle(f"Deviation from mean for {module}")
    for i, ax in enumerate(axes.reshape(-1)):
        if i == ndets:
            break
        ms = 1
        ax.plot(times, means[:, i, 0]-means[:, i, 0].mean(), "o",
                label="121", alpha=0.2, markersize=ms)
        ax.plot(times, means[:, i, 1]-means[:, i, 1].mean(), "o",
                label="344", alpha=0.2, markersize=ms)
        ax.plot(times, means[:, i, 2]-means[:, i, 2].mean(), "o",
                label="1333", alpha=0.2, markersize=ms)

        # ax.axvline(nruns[0], color="k", linestyle="--")
        if i == 0:
            ax.legend()
        ax.text(0.7, 0.8, f'ch{i}', transform=ax.transAxes)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    fig.text(0.5, 0.04, 'runs', ha='center')
    fig.text(0.04, 0.5, 'position - mean(position) in keV', va='center',
             rotation='vertical')
    fig.savefig(figdir / f"diff_{module}.jpeg")

    print(f"Cannot estimate for items {np.argwhere(np.isnan(means))}")

    # # plot of the runs where fit did not succeed
    # fix, ax = plt.subplots()
    # det = 12
    # for run in [0, 7, 128, 163, 185, 196]:
    #     y = runs[run][:, det]
    #     x = np.arange(len(y))*pars[det, 1] + pars[det, 0]
    #     x = x*pars2[det, 1] + pars2[det, 0]
    #     x = x*pars3[det, 1] + pars3[det, 0]
    #     ax.plot(x, y, alpha=0.5)

    # plot of the runs where fit did not succeed
    if module == "mod1":
        fig, ax = plt.subplots()
        det = 2
        for run in [415, 416, 670, 676]:
            y = runs[run][:, det]
            x = np.arange(len(y))*pars[det, 1] + pars[det, 0]
            x = x*pars2[det, 1] + pars2[det, 0]
            x = x*pars3[det, 1] + pars3[det, 0]
            fmt = "-" if run > 420 else "--"
            ax.plot(x, y, fmt, alpha=0.5, label=f"run={run}")
            ax.set_xlim(1250, 1400)
            ax.set_ylim(0, 4e3)
        ax.legend()
        fig.savefig(figdir / f"fig_special_{module}.jpeg")

    plt.show()


if __name__ == "__main__":
    analysis_per_module("mod0")
    analysis_per_module("mod1")


