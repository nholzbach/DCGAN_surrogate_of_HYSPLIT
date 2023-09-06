import matplotlib.pyplot as plt
import xarray as xr
import os, sys

def main(argv):
    filename = argv[1]

    ds_disk = xr.open_dataset(filename)
    print("file opened:", filename)

    mean_data = {}
    mean_data['time'] = times
    for var_name in variables:
        fig, ax = plt.subplots()
        means =[]
        for time in times:
            var = ds_disk[var_name].sel(time=time)
            var_mean = var.mean(dim='x', skipna=True).mean()
            means.append(var_mean.values)
        mean_data[var_name] = means
    print("means extracted")

    input_data = pd.DataFrame(mean_data)

    dfs = []
    sources = ['Glassport', 'Harrison', 'Lawrenceville', 'Liberty', 'NorthBraddock']
    for source in sources:
        print("getting source data for", source)
        source_data = pd.read_csv(f'../zips/sensor_data/{source}_emission_filled.csv')
        source_data = source_data.rename(columns={'Unnamed: 0': 'time', 'emission':source})
        source_data['time'] = pd.to_datetime(source_data['time']).dt.strftime('%Y-%m-%d %H:%M:%S').astype('datetime64[ns]')

        dfs.append(source_data.set_index('time'))

    # Combine the DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, axis=1).reset_index()
    merged = pd.merge(input_data, combined_df, on='time')

if __name__ == "__main__":
    main(sys.argv)
