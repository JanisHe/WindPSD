import os
import obspy

from pathlib import Path

import numpy as np


def convert2utc(array):
    array_out = []
    for i, value in enumerate(array):
        if isinstance(value, np.datetime64) is True:
            array_out.append(obspy.UTCDateTime(str(value)))
        else:
            array_out.append(obspy.UTCDateTime(value))

    return array_out


def check_parameters(parameters: dict) -> dict:
    """

    :param parameters:
    :return:
    """
    # Converting start- and end time to obspy UTCDateTime format
    if isinstance(parameters["starttime"], str):
        parameters["starttime"] = obspy.UTCDateTime(parameters["starttime"])

    if isinstance(parameters["endtime"], str):
        parameters["endtime"] = obspy.UTCDateTime(parameters["endtime"])

    # Check limits of start- and end time
    if parameters["endtime"] <= parameters["starttime"]:
        msg = f"'endtime' ({parameters['endtime']}) is before 'starttime' ({parameters['starttime']})."
        raise ValueError(msg)

    # Check if SDS path is available
    if not os.path.isdir(parameters["sds_path"]):
        msg = f"{parameters['sds_path']} is not a directory."
        raise ValueError(msg)

    # Check all limits
    if parameters["windmin"] >= parameters["windmax"]:
        msg = f"'windmin' ({parameters['windmin']}) is greater or equal than 'windmax' ({parameters['windmax']})."
        raise ValueError(msg)

    if parameters["windmin"] + parameters["binsize"] >= parameters["windmax"]:
        msg = "'windmin' + 'binsize' are greater or equal than 'windmax'."
        raise ValueError(msg)

    if parameters["x_min"] >= parameters["x_max"]:
        msg = "'x_min' is greater or equal than 'x_max'."
        raise ValueError(msg)

    if parameters["y_min"] >= parameters["y_max"]:
        msg = "'y_min' is greater or equal than 'y_max'."
        raise ValueError(msg)

    if parameters["numquantile"] not in range(0, 101):
        msg = "'numquantile' is not in range [0, 100]."
        raise ValueError(msg)

    if parameters.get("save_plot"):
        if not os.path.isdir(Path(parameters["save_plot"]).parent):
            msg = f"Saving the final plot will fail, since {Path(parameters['save_plot']).parent} is not a valid directory."
            raise ValueError(msg)

    return parameters


def wind_index(bins: np.array, windspeed: float) -> int:
    try:
        idx = np.where(bins > windspeed)[0][0]
    except IndexError:
        idx = len(bins)

    return idx - 1
