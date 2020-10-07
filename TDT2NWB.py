# Imports
from pynwb import NWBFile
import tdt
import numpy as np
import pandas as pd
from pynwb.base import TimeSeries
from argparse import ArgumentParser
from pynwb.ecephys import ElectricalSeries
from pynwb import NWBHDF5IO


def read_electrodes(electrode_table_path):
    df = pd.read_excel(electrode_table_path, sheet_name=0)
    return df


def convert_to_ts(tdt_stream, name, comments, description, unit="NA"):
    return TimeSeries(
        name=name,
        data=tdt_stream.data.T,
        rate=tdt_stream.fs,
        comments=comments,
        description=description,
        unit=unit
    )


def write_nwb(out_name, nwb):
    io = NWBHDF5IO(out_name, mode="w")
    io.write(nwb)
    io.close()


def main(block_path, electrode_table_path, subject_id):
    print(block_path, electrode_table_path, subject_id)
    tdt_data = tdt.read_block(block_path)
    nwb = NWBFile(
        session_description="test_reading_TDT_file",  # required
        identifier=subject_id,  # required
        session_start_time=tdt_data.info.start_date,
    )
    nwb.create_device("PZ5")
    electrode_group = nwb.create_electrode_group(
        "tetrode",
        description="electrode array",
        location="brain",
        device=nwb.devices["PZ5"],
    )
    electrode_df = read_electrodes(electrode_table_path)
    nwb.add_electrode_column("label", "Freesurfer Label")
    nwb.add_electrode_column("chan_type", "type of channel (EEG/EKG/HR)")
    nwb.add_electrode_column("seizure_onset", "seizure onset zone")
    nwb.add_electrode_column("interictal_activity", "shows interictal activity")
    nwb.add_electrode_column("out", "outside the brain")
    nwb.add_electrode_column("spec", "intracranial electrode spec")
    nwb.add_electrode_column("bad", "bad_electrodes")
    for idx, row in electrode_df.iterrows():
        if "Ref" in row.Label:
            continue
        if hasattr(row, "LEPTO_coords_1"):
            nwb.add_electrode(
                id=idx,
                x=row.LEPTO_coords_1,
                y=row.LEPTO_coords_2,
                z=row.LEPTO_coords_3,
                imp=float(-1),
                location=row.FS_vol,
                label = row.FS_label,
                filtering="none",
                group=electrode_group,
                chan_type="EEG",
                seizure_onset=False,
                interictal_activity=False,
                spec=row.iloc[6],
                out=row.iloc[9],
                bad=False
            )
        else:
            nwb.add_electrode(
                id=idx,
                x=-1.,
                y=-1.,
                z=-1.,
                imp=float(-1),
                location=row.Label,
                label="none",
                filtering="none",
                group=electrode_group,
                chan_type="EEG",
                seizure_onset=False,
                interictal_activity=False,
                spec=row.iloc[6],
                out=row.iloc[9],
                bad=False
            )
    eeg_ind = 0
    for ind, name in enumerate(electrode_df.Label):
        if "Ref" in name:
            break
        eeg_ind += 1

    electrode_table_region = nwb.create_electrode_table_region(
        list(range(eeg_ind)), "EEG_electrodes"
    )

    es = ElectricalSeries(
        name="EEG Data",
        data=np.concatenate((tdt_data.streams.EEG1.data, tdt_data.streams.EEG2.data)).T,
        rate=tdt_data.streams.EEG1.fs,
        starting_time=tdt_data.streams.EEG1.start_time,
        electrodes=electrode_table_region,
    )
    nwb.add_acquisition(es)
    wav5 = convert_to_ts(tdt_data.streams.Wav5, "audio", "audio", "")
    nwb.add_acquisition(wav5)
    """
    try:
      wav6 = convert_to_ts(tdt_data.streams.Wav6, "Wav6", "audio2", "")
      nwb.add_acquisition(wav6)
    except AttributeError as e:
      print(e)
    """
    write_nwb("{}_{}.nwb".format(subject_id, tdt_data.info.blockname), nwb)


if __name__ == "__main__":
    parser = ArgumentParser("Read in paths to data")
    parser.add_argument("block_path")
    parser.add_argument("electrode_table_path")
    parser.add_argument("subject_id")
    args = parser.parse_args()
    main(args.block_path, args.electrode_table_path, args.subject_id)
