"""
Functions to read in seismic data and fetching wind speed data using meteostat.
The data are read from the year and julian day in a SeisComp Data Structure (SDS)
Modify, if you use a different data format and return the obspy stream.
"""

import os
import warnings
import datetime
import obspy

import numpy as np

from typing import Union
from meteostat import Point, Hourly
from obspy.clients.filesystem.sds import Client

from windpsd.utils import convert2utc


def get_waveforms_client(
    network: str,
    station: str,
    location: str,
    channel_code: str,
    client: obspy.clients.filesystem.sds.Client,
    starttime: obspy.UTCDateTime,
    endtime: obspy.UTCDateTime,
) -> obspy.Stream:
    """
    Reads waveform data from a given obspy SDS client.
    The function returns an obspy Stream. If no data are
    found, the stream does not contain any trace.

    :param network: Name of seismic network
    :param station: Name of seismic station
    :param location: Location code of seismic station
    :param channel_code: Channel code of seismic station, e.g. HH, EH, BH, ...
    :param client: Obspy SDS client
    :param starttime: Start time of the picking period
    :param endtime: End time of the picking period
    """
    # Read waveform data using obspy client
    try:
        stream = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=f"{channel_code}*",
            starttime=starttime,
            endtime=endtime,
        )
    except ValueError:
        return obspy.Stream()

    return stream


def get_waveforms_sds_path(
    network: str,
    station: str,
    location: str,
    channel_code: str,
    sds_path: str,
    starttime: obspy.UTCDateTime,
    endtime: obspy.UTCDateTime,
) -> obspy.Stream:
    """
    Backup function if reading waveform data from obspy SDS client does not work.
    In that case, this function tries to create an own pathname to read the
    seismic data from the SDS path. If no data are found, the stream contains
    no data. Gaps are filled by zeros (stream.merge(fill_value=0)).

    :param network: Name of seismic network
    :param station: Name of seismic station
    :param location: Location code of seismic station
    :param channel_code: Channel code of seismic station, e.g. HH, EH, BH, ...
    :param sds_path: Pathname of SDS (SeisComp Data Structure)
    :param starttime: Start time of the picking period
    :param endtime: End time of the picking period
    """
    if not os.path.isdir(sds_path):
        msg = f"Pathname {sds_path} to read waveform data does not exist."
        raise IOError(msg)

    if endtime - starttime > 86400:  # Raise error when reading data longer than one day
        msg = "Reading data longer than on day is not provided."
        raise ValueError(msg)

    sds_pathname = os.path.join(
        "{sds_path}",
        "{year}",
        "{network}",
        "{station}",
        "{channel}*",
        "{network}.{station}.{location}.{channel}*{year}.{julday}",
    )
    pathname = sds_pathname.format(
        sds_path=sds_path,
        year=starttime.year,
        network=network,
        station=station,
        channel=channel_code,
        location=location,
        julday="{:03d}".format(
            starttime.julday
        ),  # Format julian day as string with three characters
    )

    # Read waveform data
    try:
        stream = obspy.read(
            pathname_or_url=pathname, starttime=starttime, endtime=endtime
        )
    except Exception:
        stream = obspy.Stream()  # Return empty stream if no data are found

    return stream


def get_waveforms(
    station: str,
    network: str,
    location: str,
    channel_code: str,
    sds_path: str,
    starttime: obspy.UTCDateTime,
    endtime: obspy.UTCDateTime,
):
    """
    Main function to read seismic data from a given SeisComp Data Structure (SDS)
    pathname. First the function tries to use an obspy sds client. If no data are
    found, a second function tries to read the data from the SDS path.
    The function returns an obspy stream, which has no traces if no data were found.

    :param station: Name of seismic network
    :param network: Name of seismic station
    :param location: Location code of seismic station
    :param channel_code: Channel code of seismic station, e.g. HH, EH, BH, ...
    :param sds_path: Pathname of SDS (SeisComp Data Structure)
    :param starttime: Start time of the picking period
    :param endtime: End time of the picking period
    :return:
    """
    # Try to read data from obspy client
    client = Client(sds_root=sds_path)
    stream = get_waveforms_client(
        network=network,
        station=station,
        location=location,
        channel_code=channel_code,
        client=client,
        starttime=starttime,
        endtime=endtime,
    )

    # If stream has no data (i.e. len(stream) == 0), try to read data from sds path
    if len(stream) == 0:
        stream = get_waveforms_sds_path(
            network=network,
            station=station,
            location=location,
            channel_code=channel_code,
            sds_path=sds_path,
            starttime=starttime,
            endtime=endtime,
        )

    # Print warning if no data were found
    if len(stream) == 0:
        msg = f"No data for {network}.{station}.{location}.{channel_code}* were found between {starttime} and {endtime}."
        warnings.warn(msg)

    return stream


def fetch_metostat(
    inventory: Union[str, obspy.Inventory],
    starttime: Union[obspy.UTCDateTime, datetime.datetime],
    endtime: Union[obspy.UTCDateTime, datetime.datetime],
    network: str,
    station: str,
    location: str,
    channel: str,
    elevation: int = 20,
) -> (np.array, np.array):
    """
    Loading hourly wind speed data at a given seismological station.
    The function returns to numpy arrays, where the first contains the datetimes
    and the second the wind speed data in m/s.

    :param inventory: Inventory file of seismological station
    :param starttime: Start time of period
    :param endtime: End time of period
    :param network: Name of seismic network
    :param station: Name of seismic station
    :param location: Location of seismic station
    :param channel: Channel name, e.g. HHZ, HHN, HHE
    :param elevation: Elevation in m above ground to load wind speed data with meteostat
    :return:
    """
    # Read inventory from seismic station
    if isinstance(inventory, str):
        inventory = obspy.read_inventory(inventory)

    # Create datetime object from start- and endtime
    if isinstance(starttime, obspy.UTCDateTime):
        starttime = starttime.datetime
    if isinstance(endtime, obspy.UTCDateTime):
        endtime = endtime.datetime

    # Select correct location of station from inventory
    station_metadata = inventory.get_channel_metadata(
        seed_id=f"{network}.{station}.{location}.{channel[:3]}"
    )

    # Fetch windspeed data from meteostat
    location = Point(
        lat=station_metadata["latitude"],
        lon=station_metadata["longitude"],
        alt=elevation,
    )  # Create location for meteostat
    data = Hourly(
        loc=location, start=starttime, end=endtime
    )  # Read hourly data from meteostat
    data = data.fetch()  # Fetch data

    # Convert to arrays
    windspeed = data["wspd"] / 3.6
    datetimes = data["wspd"].axes
    datetimes = convert2utc(
        datetimes[0].values
    )  # Convert meteostat datetimes to obspy UTCDateTime format

    return np.array(datetimes), np.array(windspeed.values)
