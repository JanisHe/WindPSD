import os
import sys
import yaml
import copy
import obspy
import matplotlib

if "DISPLAY" not in os.environ:
    matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from typing import Union
from tqdm import tqdm

from obspy.signal import PPSD

from windpsd.utils import check_parameters, wind_index
from windpsd.data_io import get_waveforms, fetch_metostat


def main(parfile: Union[str, dict], verbose: bool = True):
    # Read parameter file
    if isinstance(parfile, str):
        with open(parfile, "r") as f:
            parameters = yaml.safe_load(f)
    elif isinstance(parfile, dict):
        parameters = parfile
    else:
        msg = f"parfile must be either of type 'str' or 'dict' but is of type {type(parfile)}."
        raise ValueError(msg)

    # Check all parameters in parfile
    parameters = check_parameters(parameters=parameters)

    # Print parameters if verbose is true
    if verbose:
        print("Loaded parameters from parfile", flush=True)
        for key, value in parameters.items():
            print(f"{key}: {value}", flush=True)
        print(flush=True)

    # Build windspeed and rotation bins
    bins = np.arange(
        start=parameters["windmin"],
        stop=parameters["windmax"] + parameters["binsize"],
        step=parameters["binsize"],
    )

    # Read station inventory to remove response information
    inv = obspy.read_inventory(parameters["response_file"])

    # Get waveform from first date and create dummy PPSDs for each wind bin
    dummy_starttime = copy.copy(parameters["starttime"])
    dummy_waveform = obspy.Stream()
    dummy_count = 0
    while len(dummy_waveform) == 0:
        dummy_waveform = get_waveforms(
            network=parameters["network"],
            station=parameters["station"],
            location=parameters["location"],
            channel_code=parameters["channel"],
            sds_path=parameters["sds_path"],
            starttime=dummy_starttime,
            endtime=dummy_starttime + 3600,  # Read one hour data
        )
        dummy_starttime = dummy_starttime + 3600  # Update dummy start time by one hour
        dummy_count += 1

        # Raise Error if no data are found
        if dummy_count > 200:
            msg = (
                f"Did not find seismic data in the period {parameters['starttime']} till {dummy_starttime} to "
                f"initialize PPSD class. Please select a different time period."
            )
            raise ValueError(msg)

    # Create PPSD for each wind bin
    ppsd_bins = []
    for idx in range(len(bins)):
        ppsd_bins.append(
            PPSD(
                stats=dummy_waveform[0].stats,
                metadata=inv,
                db_bins=(-200, -50, 0.1),
                period_limits=(dummy_waveform[0].stats.delta / 2.0, 2),
                ppsd_length=3550,  # Length of PPSD in seconds. Here 1 hour data
                period_smoothing_width_octaves=parameters["smooth_width"],
                period_step_octaves=1.0 / 128,
            )
        )

    # Reading wind speed data and datetimes of wind speed to read seismic data for same time period
    windspeed_datetimes, windspeed = fetch_metostat(
        inventory=inv,
        starttime=parameters["starttime"],
        endtime=parameters["endtime"],
        network=parameters["network"],
        station=parameters["station"],
        location=parameters["location"],
        channel=parameters["channel"],
        elevation=parameters.get(
            "elevation", 20
        ),  # Read elevation above ground from parfile, if not set use 20 m
    )

    # Loop over each datetime from wind speed, read seismic data and add seismic data to PPSD bin
    with tqdm(
        total=len(windspeed_datetimes) - 1,
        desc="Processing seismic data",
        ncols=100,
        bar_format="{l_bar}{bar} [Elapsed time: {elapsed} {postfix}]",
    ) as pbar:
        for idx in range(1, len(windspeed_datetimes)):
            waveform = get_waveforms(
                network=parameters["network"],
                station=parameters["station"],
                location=parameters["location"],
                channel_code=parameters["channel"],
                sds_path=parameters["sds_path"],
                starttime=windspeed_datetimes[idx - 1],
                endtime=windspeed_datetimes[idx],
            )

            # Find bin for windspeed to add waveform to PPSD bins
            windidx = wind_index(bins=bins, windspeed=windspeed[idx])

            # Add waveform to PPSD bins
            ppsd_bins[windidx].add(stream=waveform)

            # Update progress bar
            pbar.update()

    # Create plot
    # Define colorbar and its limits
    norm = matplotlib.colors.Normalize(vmin=np.min(bins), vmax=np.max(bins))
    c_m = plt.get_cmap(parameters["colorbar"], len(bins))
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    # Creating figure canvas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Try to plot each PSD curve of each PPSD bin
    for idx in range(len(bins)):
        try:
            psd = ppsd_bins[idx].get_percentile(
                percentile=parameters["numquantile"]
            )  # Get PSD percentile from wind bins
            # Obtain frequency and spectra from PPSD
            freq = 1.0 / psd[0]
            if parameters["decibel"] is True:
                amplitude = psd[1]
            elif parameters["decibel"] is False:
                amplitude = 10 ** (psd[1] / 10) * 1e9**2  # Amplitude in nm/s**2

            # Plot frequency vs. amplitude
            if parameters["decibel"] is True:
                ax.plot(
                    freq,
                    amplitude,
                    label=f"{str(bins[idx])} m/s",
                    color=s_m.to_rgba(bins[idx]),
                    lw=1.5,
                )
            elif parameters["decibel"] is False:
                ax.semilogy(
                    freq,
                    amplitude,
                    label=f"{str(bins[idx])} m/s",
                    color=s_m.to_rgba(bins[idx]),
                    lw=1.5,
                )
        except Exception as e:
            print(f"{e} for bin {bins[idx]}", flush=True)

    # Set ticks for colorbar
    zticks = []
    for idx in range(len(bins)):
        if idx == 0:
            zticks.append("<= " + str(bins[idx + 1]))
        elif idx == len(bins) - 1:
            zticks.append("> " + str(bins[idx]))
        elif 0 < idx < len(bins) - 1:
            zticks.append(str(bins[idx]) + " - " + str(bins[idx + 1]))

    # Labeling axes and colorbar
    ax.set_xlim([parameters["x_min"], parameters["x_max"]])
    ax.set_ylim([parameters["y_min"], parameters["y_max"]])
    ax.grid(color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Frequency (Hz)")
    if parameters["decibel"] is True:
        ax.set_ylabel("PSD (m$^2$/s) (dB)")
    elif parameters["decobel"] is False:
        ax.set_ylabel("PSD (nm$^2$/s)")
    name = f"{parameters['network']}.{parameters['station']}.{parameters['location']}.{parameters['channel']}"
    ax.set_title(
        f"{name}  {str(parameters['starttime'])[:10]} -- {str(parameters['endtime'])[:10]}"
    )
    cbar = plt.colorbar(s_m, ax=ax, spacing="uniform", extend="neither")
    ticklocs = []
    distlocs = (max(bins) - min(bins)) / len(bins)  # Distance between each tick
    ticklocs.append(min(bins) + distlocs / 2)
    for idx in range(len(bins) - 1):
        ticklocs.append(ticklocs[idx] + distlocs)
    cbar.set_ticks(ticklocs)
    cbar.ax.set_yticklabels(zticks)
    cbar.set_label("Wind speed (m/s)")

    # Save or show plot
    if parameters.get("save_plot"):
        plt.savefig(parameters["save_plot"])
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        parfile = "../parfiles/parfile.yml"
    elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
        msg = "The given file {} does not exist. Perhaps take the full path of the file.".format(
            sys.argv[1]
        )
        raise FileNotFoundError(msg)
    else:
        parfile = sys.argv[1]

    # Start to pick phases from parfile
    main(parfile=parfile)
