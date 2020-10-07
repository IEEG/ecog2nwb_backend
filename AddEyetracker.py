from pynwb.base import TimeSeries
import argparse
import numpy as np
import h5py
from pynwb import NWBHDF5IO


def get_description():
    return "Data is organized in (t x m) format where n=time and m=measure. The 13 measures are, in order: \
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
           Validity"

def read_eye_tracker_data(eyetracker_path):
    left_trials = []
    right_trials = []
    sum_ts = 0
    with h5py.File(eyetracker_path, "r") as f:
        #left_trials = [np.array(f[key[0]]).T for key in list(f["Eye_fv"]["LEC_fv"])]
        #right_trials = [np.array(f[key[0]]).T for key in list(f["Eye_fv"]["REC_fv"])]
        left_trials = np.array(f["Eye_movie"]["leftEye"]).T
        right_trials = np.array(f["Eye_movie"]["rightEye"]).T
    left_eye_tracking = np.concatenate(left_trials, axis=0)
    right_eye_tracking = np.concatenate(right_trials, axis=0)
    lts = TimeSeries(
        "Left Eye Tracking",
        data=left_eye_tracking,
        unit=None,
        rate=300.0,
        comments="Eye tracking from the left eye using the Tobii eye tracker",
        description=get_description()
    )
    rts = TimeSeries(
        "Right Eye Tracking",
        data=right_eye_tracking,
        unit=None,
        rate=300.0,
        comments="Eye tracking from the right eye using the Tobii eye tracker",
        description=get_description()
    )
    return lts, rts


def main(nwb, eyetracking_path):
    io = NWBHDF5IO(nwb, mode="a")
    nwbfile = io.read()
    left_et_data, right_et_data = read_eye_tracker_data(eyetracking_path)
    nwbfile.add_acquisition(left_et_data)
    nwbfile.add_acquisition(right_et_data)
    io.write(nwbfile)
    io.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("parses the Eyetracking data and adds to NWB file")
    parser.add_argument("nwbfile_path")
    parser.add_argument("eye_tracking_path")
    args = parser.parse_args()
    main(args.nwbfile_path, args.eye_tracking_path)


