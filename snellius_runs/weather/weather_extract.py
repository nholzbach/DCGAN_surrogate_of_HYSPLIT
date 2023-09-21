import os, sys
import pandas as pd
import dataclasses
import datetime
import s3fs
import xarray as xr
import numpy as np
import numcodecs as ncd
import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import concurrent.futures
import matplotlib.pyplot as plt

# FUNCTIONS
# this is all the original code from utah site are fundamental functions
@dataclasses.dataclass
class ZarrId:
    run_hour: datetime.datetime
    level_type: str
    var_level: str
    var_name: str
    model_type: str

    def format_chunk_id(self, chunk_id):
        if self.model_type == "fcst":
            # Extra id part since forecasts have an additional (time) dimension
            return "0." + str(chunk_id)
        else:
            return chunk_id

# build a class for the extraction
class HRRR_extraction():
    def __init__(self, yr, mo, day, length):
        self.yr = yr
        self.mo = mo
        self.day = day
        self.length = length

        self.lat_top = 40.5746
        self.lat_bottom = 40.2393
        self.lon_top = -79.7231
        self.lon_bottom = -80.1676

        self.start = datetime.datetime(yr, mo, day, 0)
        self.end = self.start + datetime.timedelta(days=length)
        # hourly date range
        self.times = pd.date_range(start=self.start, end=self.end, freq='H')

        self.s3 = boto3.resource(service_name='s3', region_name='us-west-1', config=Config(signature_version=UNSIGNED))
        self.chunk_id = "4.9" # PA is in here
        self.fs = s3fs.S3FileSystem(anon=True)
        self.chunk_index = xr.open_zarr(s3fs.S3Map("s3://hrrrzarr/grid/HRRR_chunk_index.zarr", s3=self.fs))

        self.variables = ['UGRD', 'VGRD', 'TMP', 'PRES', 'SFCR', 'RH', 'TCDC']
        self.var_levels = ['10m_above_ground', "10m_above_ground", '1000mb', 'surface', 'surface', '2m_above_ground', 'entire_atmosphere']

    def create_s3_group_url(self, zarr_id, prefix=True):
        url = "s3://hrrrzarr/" if prefix else "" # Skip when using boto3
        url += zarr_id.run_hour.strftime(
            f"{zarr_id.level_type}/%Y%m%d/%Y%m%d_%Hz_{zarr_id.model_type}.zarr/")
        url += f"{zarr_id.var_level}/{zarr_id.var_name}"
        return url
    def create_s3_subgroup_url(self, zarr_id, prefix=True):
        url = self.create_s3_group_url(zarr_id, prefix)
        url += f"/{zarr_id.var_level}"
        return url

    def create_s3_chunk_url(self, zarr_id, chunk_id, prefix=False):
        url = self.create_s3_subgroup_url(zarr_id, prefix)
        url += f"/{zarr_id.var_name}/{zarr_id.format_chunk_id(chunk_id)}"
        return url

    def decompress_chunk(self, zarr_id, compressed_data):
        buffer = ncd.blosc.decompress(compressed_data)

        dtype = "<f2"
        if zarr_id.var_level == "surface" and zarr_id.var_name == "PRES":
            dtype = "<f4"

        chunk = np.frombuffer(buffer, dtype=dtype)

        if zarr_id.model_type == "anl":
            print(len(chunk), zarr_id.var_name, zarr_id.run_hour)
            if len(chunk) != 22500:
                print(f"chunk is not the right size for {zarr_id.var_name, zarr_id.run_hour}, so making it zeros")
                data_array = np.zeros((150, 150))

            else:
                data_array = np.reshape(chunk, (150, 150))
        else:
            entry_size = 22500
            data_array = np.reshape(chunk, (len(chunk)//entry_size, 150, 150))

        return data_array


    def check_boundaries(self, data):
        return (self.lat_bottom < data.latitude) & (data.latitude < self.lat_top) & (
            self.lon_bottom < data.longitude) & (data.longitude < self.lon_top)

    def retrieve_object(self, s3_url):
        obj = self.s3.Object('hrrrzarr', s3_url)
        return obj.get()['Body'].read()

    def get_unique(self, data):
        # We have to implement our own "unique" logic since missing values are NaN (a float) and the rest are string
        data = data.fillna(None).values.flatten()
        data = data[data != None]
        return np.unique(data)

    def get_chunk(self, zarr_id, chunk_id):
        # retrieve data as before
        compressed_data = self.retrieve_object(self.create_s3_chunk_url(zarr_id, chunk_id))
        chunk_data = self.decompress_chunk(zarr_id, compressed_data)
        # combine retrieved data with the chunk grid
        chunk_xarray = self.chunk_index.where(lambda x: x.chunk_id == chunk_id, drop=True)
        dimensions = ("y", "x") if zarr_id.model_type == "anl" else ("time", "y", "x")
        chunk_xarray[zarr_id.var_name] = (dimensions, chunk_data)

        return chunk_xarray

    def get_chunks_combined(self, zarr_id, chunk_ids):
        chunks = [self.get_chunk(zarr_id, chunk_id) for chunk_id in chunk_ids]
        return xr.merge(chunks)


    def get_data(self, zarr_ids, chunk_ids, is_forecast):
        datasets = []
        for zarr_id in zarr_ids:
            try:
                data = self.get_chunk(zarr_id,chunk_ids[0])
            except Exception as e:
                print(f"Skipping chunk with zarr_id={zarr_id}, due to error: {e}")
                # make empty xarray
                x = np.linspace(1.352e6, 1.799e6, 150)
                y = np.linspace(2.127e5, 6.597e5, 150)
                empty = xr.DataArray(0.0, dims=('x', 'y'), coords={'x': x, 'y': y})
                chunkar = xr.DataArray('4.9', dims=('x', 'y'), coords={'x': x, 'y': y})
                data = xr.Dataset({
                    'chunk_id': chunkar,
                    'chunk_x': empty,
                    'chunk_y': empty,
                    'in_chunk_x': empty,
                    'in_chunk_y': empty,
                    'index_x': empty,
                    'index_y': empty,
                    'latitude': empty,
                    'longitude': empty,
                    f'{zarr_id.var_name}': empty,
                })
            new_time_dimension = "run_time" if is_forecast else "time"
            data[new_time_dimension] = zarr_id.run_hour
            datasets.append(data)
        ds = xr.concat(datasets, dim=new_time_dimension, combine_attrs="override")
        return ds

    def steps_per_var(self, var_name, var_level):
        zarr_id = ZarrId(
                        run_hour=datetime.datetime(self.yr, self.mo, self.day, 0),
                        level_type="sfc",
                        var_level=var_level,
                        var_name=var_name,
                        model_type="anl"
                        )
        area = self.chunk_index.where(self.check_boundaries, drop=True)
        chunk_ids = self.get_unique(area.chunk_id)
        data = self.get_chunk(zarr_id,chunk_ids[0])
        data = data.where(self.check_boundaries, drop=True)
        data = data.drop_vars(['chunk_id', 'chunk_y', 'chunk_x', 'in_chunk_y', 'in_chunk_x', 'index_x', 'index_y'])
        zarr_ids = [dataclasses.replace(zarr_id, run_hour=time) for time in self.times]
        print("now working on getting multiple dates for ", var_name)
        multiple_dates = self.get_data(zarr_ids, chunk_ids, False)
        print("all dates collected for ", var_name)
        return multiple_dates

    def process_var(self, var_name, var_level):
        print(var_name, var_level)
        return self.steps_per_var(var_name, var_level)


    def run(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self.process_var, var_name, var_level) for var_name, var_level in zip(self.variables, self.var_levels)]

        all = [future.result() for future in results]
        print("all array has been constructed")
        return all

    def combine_and_save(self, all):
        combined_data = {}
        # Extract variable and combine into one dataset
        for i, var in enumerate(self.variables):
            dataset = all[i]
            data_array = dataset[var]
            combined_data[var] = data_array
            combined_data[var] = combined_data[var].astype('float32')

            if i == 0:
                lat = dataset.latitude
                lon = dataset.longitude
                combined_data['latitude'] = lat.astype('float32')
                combined_data['longitude'] = lon.astype('float32')
        combined_dataset = xr.Dataset(combined_data)
        print("Data sets combined into a dataset, now saving to netcdf")

        combined_dataset.to_netcdf(f'nc/combined_data_{self.yr}_{self.mo}_{self.length}.nc', engine='h5netcdf')
        print("DONE EXTRACTING!!")

    def make_input_data(self, filepattern):
        ds_disk = xr.open_mfdataset(filepattern, combine='nested', concat_dim='time')

        print("files opened:", filepattern)

        print(ds_disk.time)
        month_data = {}

        # changing resolution of grid to 15x15
        print(len(self.times))
        ds_coarse = ds_disk.coarsen(x=10, y=10, boundary='exact').mean()

        for i, time in enumerate(self.times):
            data ={}
            for var_name in self.variables:

                var = ds_coarse[var_name].isel(time=i)
                var_coarse = var.values
                data[var_name] = var_coarse

            month_data[time] = data

        month_df = pd.DataFrame(month_data)
        # flip so that time is on the
        month_df = month_df.T
        month_df.to_csv(f'input_data/weather_input_data_{self.yr}_{self.mo}.csv')
        print("saved weather data to csv, DONE")


def main(args):
    yr = int(args[1])
    mo = int(args[2])
    day = int(args[3])
    length = int(args[4])

    test = HRRR_extraction(yr, mo, day, length)
    all = test.run()
    test.combine_and_save(all)
    test.make_input_data(f'nc/combined_data_{yr}_{mo}_*.nc')

if __name__ == "__main__":
    main(sys.argv)

