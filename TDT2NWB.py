# Imports
from pynwb import NWBFile
import tdt
import numpy as np
import pandas as pd
from pynwb.base import TimeSeries
from argparse import ArgumentParser
from pynwb.ecephys import ElectricalSeries
import h5py
from pynwb import NWBHDF5IO


def read_electrodes(electrode_table_path):
    df = pd.read_excel(electrode_table_path, sheet_name="TDT")
    return df


def convert_to_ts(tdt_stream, name, comments, description, unit="NA"):
    return TimeSeries(
        name=name,
        data=tdt_stream.data,
        rate=tdt_stream.fs,
        comments=comments,
        description=description,
        unit=unit
    )


def read_eye_tracker_data(eyetracker_path):
    timing = []
    left_trials = []
    right_trials = []
    sum_ts = 0
    with h5py.File(eyetracker_path, "r") as f:
        for key in list(f["Eye_fv"]["TS_fv"]):
            timing_trial = np.array(f[key[0]])
            timing.append(timing_trial.T)
        for key in list(f["Eye_fv"]["LEC_fv"]):

            array_trials = np.array(f[key[0]])
            left_trials.append(array_trials.T)

        for key in list(f["Eye_fv"]["REC_fv"]):
            array_trials = np.array(f[key[0]])
            right_trials.append(array_trials.T)
            sum_ts += array_trials.shape[1]
    left_eye_tracking = np.concatenate(left_trials, axis=0)
    right_eye_tracking = np.concatenate(right_trials, axis=0)
    lts = TimeSeries(
        "Left Eye Tracking",
        data=left_eye_tracking,
        unit=None,
        rate=300.0,
        comments="Eye tracking from the left eye using the Tobii eye tracker",
        description="Data is organized in (t x m) format where n=time and m=measure. The 13 measures are, in order: \
           EyePosition3d.x \
           EyePosition3d.y \
           EyePosition3d.z \
           EyePosition3dRelative.x \
           EyePosition3dRelative.y \
           EyePosition3dRelative.z \
           GazePoint2d.x \
           GazePoint2d.y \
           GazePoint3d.x \
           GazePoint3d.y \
           GazePoint3d.z \
           PupilDiameter \
           Validity",
    )
    rts = TimeSeries(
        "Right Eye Tracking",
        data=right_eye_tracking,
        unit=None,
        rate=300.0,
        comments="Eye tracking from the right eye using the Tobii eye tracker",
        description="Data is organized in (t x m) format where n=time and m=measure. The 13 measures are, in order: \
           EyePosition3d.x \
           EyePosition3d.y \
           EyePosition3d.z \
           EyePosition3dRelative.x \
           EyePosition3dRelative.y \
           EyePosition3dRelative.z \
           GazePoint2d.x \
           GazePoint2d.y \
           GazePoint3d.x \
           GazePoint3d.y \
           GazePoint3d.z \
           PupilDiameter \
           Validity",
    )
    return lts, rts


def write_nwb(out_name, nwb):
    io = NWBHDF5IO(out_name, mode="w")
    io.write(nwb)
    io.close()


def main(block_path, eyetracker_path, electrode_table_path):
    print(block_path, eyetracker_path, electrode_table_path)
    tdt_data = tdt.read_block(block_path)
    nwb = NWBFile(
        session_description="test_reading_TDT_file",  # required
        identifier="NS140_02",  # required
        session_start_time=tdt_data.info.start_date,
    )
    nwb.create_device("TDT")
    electrode_group = nwb.create_electrode_group(
        "tetrode",
        description="electrode array",
        location="brain",
        device=nwb.devices["TDT"],
    )
    electrode_df = read_electrodes(electrode_table_path)
    for idx, row in electrode_df.iterrows():
        nwb.add_electrode(
            id=idx,
            x=row.LEPTO_coords_1,
            y=row.LEPTO_coords_2,
            z=row.LEPTO_coords_3,
            imp=float(-1),
            location=row.FS_label,
            filtering="none",
            group=electrode_group,
        )
    electrode_table_region = nwb.create_electrode_table_region(
        list(range(384)), "EEG_electrodes"
    )

    es = ElectricalSeries(
        name="EEG Data",
        data=np.concatenate((tdt_data.streams.EEG1.data, tdt_data.streams.EEG2.data)),
        rate=tdt_data.streams.EEG1.fs,
        starting_time=tdt_data.streams.EEG1.start_time,
        electrodes=electrode_table_region,
    )
    nwb.add_acquisition(es)
    wav5 = convert_to_ts(tdt_data.streams.Wav5, "Wav5", "audio", "")
    nwb.add_acquisition(wav5)
    try:
      wav6 = convert_to_ts(tdt_data.streams.Wav6, "Wav6", "audio2", "")
      nwb.add_acquisition(wav6)
    except AttributeError as e:
      print(e)
    left_et_data, right_et_data = read_eye_tracker_data(eyetracker_path)
    nwb.add_acquisition(left_et_data)
    nwb.add_acquisition(right_et_data)
    write_nwb("test_nwb.nwb", nwb)


if __name__ == "__main__":
    parser = ArgumentParser("Read in paths to data")
    parser.add_argument("block_path")
    parser.add_argument("eyetracker_path")
    parser.add_argument("electrode_table_path")
    args = parser.parse_args()
    main(args.block_path, args.eyetracker_path, args.electrode_table_path)
