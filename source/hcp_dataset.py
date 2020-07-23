"""
Contains general variables and functions relating to the HCP dataset. 
Many of these functions are taken from the Neuromatch academy HCP dataset guide.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# HCP Parameters
HCP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'hcp')
# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 339
# The data have already been aggregated into ROIs from the Glasesr parcellation
N_PARCELS = 360
# The acquisition parameters for all tasks were identical
TR = 0.72    # Time resolution, in sec
# The parcels are matched across hemispheres with the same order
HEMIS = ["Right", "Left"]
# Each experiment was repeated multiple times in each subject
N_RUNS_REST = 4
N_RUNS_TASK = 2
# Time series data are organized by experiment, with each experiment
# having an LR and RL (phase-encode direction) acquistion
BOLD_NAMES = [
    "rfMRI_REST1_LR", "rfMRI_REST1_RL",
    "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR",
    "tfMRI_WM_RL", "tfMRI_WM_LR",
    "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR",
    "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR",
    "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]
# You may want to limit the subjects used during code development.
# This will use all subjects:
subjects = range(N_SUBJECTS)
#Downloading either dataset will create the regions.npy file, which contains the region name and network assignment for each parcel.
#Detailed information about the name used for each region is provided in the Supplement to Glasser et al. 2016.
#Information about the network parcellation is provided in Ji et al, 2019.
regions = np.load(f"{HCP_DIR}/regions.npy").T
region_info = dict(
    name=regions[0].tolist(),
    network=regions[1],
    myelin=regions[2].astype(np.float),
)

#@title HCP Helper functions
def get_image_ids(name):
    """Get the 1-based image indices for runs in a given experiment.

        Args:
            name (str) : Name of experiment ("rest" or name of task) to load
        Returns:
            run_ids (list of int) : Numeric ID for experiment image files

    """
    run_ids = [
        i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code
    ]
    if not run_ids:
        raise ValueError(f"Found no data for '{name}''")
    return run_ids

def load_timeseries(subject, name, runs=None, concat=True, remove_mean=True):
    """Load timeseries data for a single subject.
    
    Args:
        subject (int): 0-based subject ID to load
        name (str) : Name of experiment ("rest" or name of task) to load
        run (None or int or list of ints): 0-based run(s) of the task to load,
            or None to load all runs.
        concat (bool) : If True, concatenate multiple runs in time
        remove_mean (bool) : If True, subtract the parcel-wise mean

    Returns
        ts (n_parcel x n_tp array): Array of BOLD data values

    """
    # Get the list relative 0-based index of runs to use
    if runs is None:
        runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
    elif isinstance(runs, int):
        runs = [runs]

    # Get the first (1-based) run id for this experiment 
    offset = get_image_ids(name)[0]

    # Load each run's data
    bold_data = [
            load_single_timeseries(subject, offset + run, remove_mean) for run in runs
    ]

    # Optionally concatenate in time
    if concat:
        bold_data = np.concatenate(bold_data, axis=-1)

    return bold_data


def load_single_timeseries(subject, bold_run, remove_mean=True):
    """Load timeseries data for a single subject and single run.
    
    Args:
        subject (int): 0-based subject ID to load
        bold_run (int): 1-based run index, across all tasks
        remove_mean (bool): If True, subtract the parcel-wise mean

    Returns
        ts (n_parcel x n_timepoint array): Array of BOLD data values

    """
    bold_path = f"{HCP_DIR}/subjects/{subject}/timeseries"
    bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
    ts = np.load(f"{bold_path}/{bold_file}")
    if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
    return ts

def load_evs(subject, name, condition):
    """Load EV (explanatory variable) data for one task condition.

    Args:
        subject (int): 0-based subject ID to load
        name (str) : Name of task
        condition (str) : Name of condition

    Returns
        evs (list of dicts): A dictionary with the onset, duration, and amplitude
            of the condition for each run.

    """
    evs = []
    for id in get_image_ids(name):
        task_key = BOLD_NAMES[id - 1]
        ev_file = f"{HCP_DIR}/subjects/{subject}/EVs/{task_key}/{condition}.txt"
        ev = dict(zip(["onset", "duration", "amplitude"], np.genfromtxt(ev_file).T))
        evs.append(ev)
    return evs

def condition_frames(run_evs, skip=0):
    """Identify timepoints corresponding to a given condition in each run.

    Args:
        run_evs (list of dicts) : Onset and duration of the event, per run
        skip (int) : Ignore this many frames at the start of each trial, to account
            for hemodynamic lag

    Returns:
        frames_list (list of 1D arrays): Flat arrays of frame indices, per run

    """
    frames_list = []
    for ev in run_evs:

        # Determine when trial starts, rounded down
        start = np.floor(ev["onset"] / TR).astype(int)

        # Use trial duration to determine how many frames to include for trial
        duration = np.ceil(ev["duration"] / TR).astype(int)

        # Take the range of frames that correspond to this specific trial
        frames = [s + np.arange(skip, d) for s, d in zip(start, duration)]

        frames_list.append(np.concatenate(frames))

    return frames_list


def selective_average(timeseries_data, ev, skip=0):
    """Take the temporal mean across frames for a given condition.

    Args:
        timeseries_data (array or list of arrays): n_parcel x n_tp arrays
        ev (dict or list of dicts): Condition timing information
        skip (int) : Ignore this many frames at the start of each trial, to account
            for hemodynamic lag

    Returns:
        avg_data (1D array): Data averagted across selected image frames based
        on condition timing

    """
    # Ensure that we have lists of the same length
    if not isinstance(timeseries_data, list):
        timeseries_data = [timeseries_data]
    if not isinstance(ev, list):
        ev = [ev]
    if len(timeseries_data) != len(ev):
        raise ValueError("Length of `timeseries_data` and `ev` must match.")

    # Identify the indices of relevant frames
    frames = condition_frames(ev)

    # Select the frames from each image
    selected_data = []
    for run_data, run_frames in zip(timeseries_data, frames):
        selected_data.append(run_data[:, run_frames])

    # Take the average in each parcel
    avg_data = np.concatenate(selected_data, axis=-1).mean(axis=-1)

    return avg_data

#@title Lukas' functions
def timeseries_to_dataframe(subject, timeseries):
    """
    Returns a dataframe in which each row is a region.
    Takes a timeseries from one subject, but can be concatenated over multiple.
    
    Args:
        timeseries (2d array): n_parcel x n_tp array

    Returns:
        data (DataFrame): A dataframe in which each row is one region(voxel?)
            columns:
                subject: subject number.
                timeseries: Contains the timeseries values of this region
                regionName: name of this region
                network: the network this region belongs to
    """
    data = pd.DataFrame()
    n_runs = len(timeseries)
    runs = []
    for i, single_timeseries in enumerate(timeseries):
        new_run = np.ones(len(single_timeseries)) * i
        runs.append(new_run)
    data['run'] = np.concatenate(runs)
    data['timeseries'] = np.concatenate(timeseries, axis=0).tolist()
    data['regionName'] = np.tile(region_info['name'], n_runs)
    data['network'] = np.tile(region_info['network'], n_runs)
    data['subject'] = subject
    print(data)
    return data

def selective(timeseries_data, ev, skip=0):
    """Take the temporal mean across frames for a given condition.

    Args:
        timeseries_data (array or list of arrays): n_parcel x n_tp arrays
        ev (dict or list of dicts): Condition timing information
        skip (int) : Ignore this many frames at the start of each trial, to account
            for hemodynamic lag

    Returns:
        selected_data (2D array)

    """
    # Ensure that we have lists of the same length
    if not isinstance(timeseries_data, list):
        timeseries_data = [timeseries_data]
    if not isinstance(ev, list):
        ev = [ev]
    if len(timeseries_data) != len(ev):
        raise ValueError("Length of `timeseries_data` and `ev` must match.")

    # Identify the indices of relevant frames
    frames = condition_frames(ev)

    # Select the frames from each image
    selected_data = []
    for run_data, run_frames in zip(timeseries_data, frames):
        selected_data.append(run_data[:, run_frames])

    return selected_data

def select_trials(timeseries_data, evs, hemo_dynamic_delay=6.0):
    """ Get the closest timeseries data for each event in evs.

    Args:
        timeseries_data (array or list of arrays): n_parcel x n_tp arrays
        ev (dict or list of dicts): Condition timing information
        hemo_dynamic_delay: hemodynamic lag in seconds

    Returns:
        selected_data (2D array)

    """
    # Ensure that we have lists of the same length
    if not isinstance(timeseries_data, list):
        timeseries_data = [timeseries_data]
    if not isinstance(evs, list):
        evs = [evs]
    if len(timeseries_data) != len(evs):
        raise ValueError("Length of `timeseries_data` and `ev` must match.")
    
    # Iterate over runs
    selected_data = []
    for run in range(len(evs)):
        # Find the acquisition closest to onset of each event
        for ev in evs[run]["onset"]:
            # Trial time to acquisition index
            frame = np.round((ev + hemo_dynamic_delay) / TR).astype(int)
            selected_data.append(timeseries_data[run][:, frame])

    return selected_data
