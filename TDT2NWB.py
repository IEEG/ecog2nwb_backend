# Imports
import tdt
import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
from pynwb import NWBFile, NWBHDF5IO
from pynwb.base import TimeSeries
from pynwb.ecephys import ElectricalSeries


class tdt2nwb:
    def __init__(
        self,
        device_name="PZ5",
        electrode_table_region_name="EEG_electrodes",
        electrical_series_name="EEG Data",
        output_folder="./",
    ):
        # initialize tdt converter

        self.device = device_name
        self.electrode_table_region_name = electrode_table_region_name
        self.electrical_series_name = electrical_series_name
        self.nwb = None
        self.output_folder = output_folder

    def read_electrode_table(self, electrode_table_path):
        # read in electrode metadata table
        # assumes that the metadata table is on the first sheet of an excel spreadsheet

        df = pd.read_excel(electrode_table_path, sheet_name=0)
        return df

    def convert_to_ts(self, tdt_stream, name, comments, description, unit="NA"):
        # converts a tdt stream to a NWB time series
        return TimeSeries(
            name=name,
            data=tdt_stream.data.T,
            rate=tdt_stream.fs,
            comments=comments,
            description=description,
            unit=unit,
        )

    def write_nwb(self, out_name):
        # write out nwb file
        out_name = os.path.join(self.output_folder, out_name)
        io = NWBHDF5IO(out_name, mode="w")
        io.write(self.nwb)
        io.close()

    def add_audio(self, tdt_data):
        # convert WAV5 in tdt to audio in NWB and add as generic time series

        wav5 = self.convert_to_ts(tdt_data.streams.Wav5, "audio", "audio", "")
        self.nwb.add_acquisition(wav5)

    def add_electrical_series(self, electrode_table_region, tdt_data):
        # add EEG data as electrical series in NWB

        es = ElectricalSeries(
            name="EEG Data",
            data=np.concatenate(
                (tdt_data.streams.EEG1.data, tdt_data.streams.EEG2.data)
            ).T,
            rate=tdt_data.streams.EEG1.fs,
            starting_time=tdt_data.streams.EEG1.start_time,
            electrodes=electrode_table_region,
        )
        self.nwb.add_acquisition(es)

    def create_electrode_region(self, electrode_df):
        # create region for EEG electrodes, separates Ref from Data
        eeg_ind = 0
        for ind, name in enumerate(electrode_df.Label):
            if "Ref" in name:
                break
            eeg_ind += 1
        electrode_table_region = self.nwb.create_electrode_table_region(
            list(range(eeg_ind)), "EEG_electrodes"
        )
        return electrode_table_region

    def populate_electrode_tables(self, electrode_df, electrode_group):
        # Populates the electrode tables with metadata from the associated spreadsheet
        # This could be improved to be more robust

        self.nwb.add_electrode_column("label", "Freesurfer Label")
        self.nwb.add_electrode_column("chan_type", "type of channel (EEG/EKG/HR)")
        self.nwb.add_electrode_column("seizure_onset", "seizure onset zone")
        self.nwb.add_electrode_column(
            "interictal_activity", "shows interictal activity"
        )
        self.nwb.add_electrode_column("out", "outside the brain")
        self.nwb.add_electrode_column("spec", "intracranial electrode spec")
        self.nwb.add_electrode_column("bad", "bad_electrodes")
        for idx, row in electrode_df.iterrows():
            if "Ref" in row.Label:
                continue
            if hasattr(row, "LEPTO_coords_1"):
                self.nwb.add_electrode(
                    id=idx,
                    x=row.LEPTO_coords_1,
                    y=row.LEPTO_coords_2,
                    z=row.LEPTO_coords_3,
                    imp=float(-1),
                    location=row.FS_vol,
                    label=row.FS_label,
                    filtering="none",
                    group=electrode_group,
                    chan_type="EEG",
                    seizure_onset=False,
                    interictal_activity=False,
                    spec=row.iloc[6],
                    out=row.iloc[9],
                    bad=False,
                )
            else:
                raise ValueError("No coordinates found in metadata")

    def create_electrode_devices(self):
        # Create and briefly describe devices used. Could be improved with more detail
        self.nwb.create_device(self.device)
        electrode_group = self.nwb.create_electrode_group(
            "tetrode",
            description="electrode array",
            location="brain",
            device=self.nwb.devices[self.device],
        )
        return electrode_group

    def run_conversion(
        self,
        block_path,
        electrode_table_path,
        session_id,
        session_description="test read",
    ):
        # Converts TDT block to NWB data

        # read in TDT file
        print("Reading {} at {}".format(session_id, block_path))
        tdt_data = tdt.read_block(block_path)

        # initialize nwb file, requires a session description, id, and start time
        self.nwb = NWBFile(
            session_description=session_description,  # required
            identifier=session_id,  # required
            session_start_time=tdt_data.info.start_date,
        )
        electrode_group = self.create_electrode_devices()
        electrode_df = self.read_electrode_table(electrode_table_path)
        self.populate_electrode_tables(electrode_df, electrode_group)
        electrode_table_region = self.create_electrode_region(electrode_df)
        self.add_electrical_series(electrode_table_region, tdt_data)
        self.add_audio(tdt_data)
        self.write_nwb("{}.nwb".format(session_id))

    def __call__(self, bp, etp, si):
        self.run_conversion(bp, etp, si)


if __name__ == "__main__":
    parser = ArgumentParser("Read in paths to data")
    parser.add_argument("block_path")
    parser.add_argument("electrode_table_path")
    parser.add_argument("session_id")
    parser.add_argument("--output_folder", nargs="?", default="./")
    args = parser.parse_args()
    tn = tdt2nwb(output_folder=args.output_folder)
    tn(args.block_path, args.electrode_table_path, args.session_id)
