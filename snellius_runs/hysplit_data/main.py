"""
The main script for processing the plume visualization videos
(copy and modify this script for your own needs)
"""


import sys, os, time
import pandas as pd
from datetime import date
from datetime import timedelta
import shutil
from multiprocessing.dummy import Pool
from cached_hysplit_run_lib_edit import DispersionSource
from automate_plume_viz_edit import get_time_range_list, simulate_worker

PATH = "/home/nholzbach/"
current_path = "/home/nholzbach/hysplit_data/"

# this gives info that is used in the main script but we don't want to use it?
def generate_earthtime_data(date_list, prefix="banana_",
        add_smell=False, lat="40.42532", lng="-79.91643", zoom="9.233", credits ="CREATE Lab",  category ="Plume Conc", name_prefix= "PARDUMP ", video_start_delay_hrs=0):
    print("Generate EarthTime data...")

    df_layer, df_share_url, df_img_url, file_name, start_d, end_d = None, None, None, None, None, None
    sd, ed = date_list[0], date_list[1]


    # dl, ds, di, fn = generate_metadata(sd, ed, video_start_delay_hrs=video_start_delay_hrs,
            # url_partition=url_partition, img_size=img_size, redo=redo, prefix=prefix,
            # add_smell=add_smell, lat=lat, lng=lng, zoom=zoom, credits=credits,
            # category=category, name_prefix=name_prefix, file_path=bin_url)

    df_template = pd.read_csv("data/earth_time_layer_template.csv")
    dl = pd.concat([df_template]*len(sd), ignore_index=True)
    # vid_start_d = start_d + pd.DateOffset(hours=video_start_delay_hrs)
    start_d_utc = sd.tz_convert("UTC")
    end_d_utc = ed.tz_convert("UTC")
    fn = prefix + start_d_utc.strftime("%Y%m%d%H%M") + "_" + end_d_utc.strftime("%Y%m%d%H%M")
    dl["Start date"] = start_d_utc.strftime("%Y%m%d%H%M%S")
    dl["End date"] = end_d_utc.strftime("%Y%m%d%H%M%S")
    dl["Share link identifier"] = file_name
    dl["Name"] = name_prefix + start_d_utc.strftime("%Y-%m-%d")
    # df_layer["URL"] = file_path + file_name + ".bin"
    dl["Category"] = category
    dl["Credits"] = credits


    if df_layer is None:
        df_layer, file_name, start_d, end_d = dl, fn, sd, ed
    else:
        df_layer = pd.concat([df_layer, dl], ignore_index=True)
        # df_share_url = pd.concat([df_share_url, ds], ignore_index=True)
        # df_img_url = pd.concat([df_img_url, di], ignore_index=True)
        file_name = file_name.union(fn)
        start_d = start_d.union(sd)
        end_d = end_d.union(ed)

    # Save rows of EarthTime CSV layers to a file
    p = "data/earth_time_layer.csv"
    df_layer.to_csv(p, index=False)
    os.chmod(p, 0o777)

    # Save rows of share urls to a file
    # p = "data/earth_time_share_urls.csv"
    # df_share_url.to_csv(p, index=False)
    # os.chmod(p, 0o777)

    # Save rows of thumbnail server urls to a file
    # p = "data/earth_time_thumbnail_urls.csv"
    # df_img_url.to_csv(p, index=False)
    # os.chmod(p, 0o777)

    return (start_d, end_d, file_name)


def run_hysplit(sources, bin_root, start_d, end_d, file_name, bin_url=None, num_workers=4, use_forecast=False):
    print("Run Hysplit model...")
    print("Using num workers: %s" % num_workers)

    # Prepare the list of dates for running the simulation
    start_time_eastern_all = start_d.strftime("%Y-%m-%d-%H:%M").values

    # Prepare the list of file names
    bin_file_all = bin_root + file_name.values + ".bin"

    # Prepare the list of URLs for checking if the file exists in the remote server
    # if bin_url is None:
    #     bin_url_all = [None]*len(file_name.values)
    # else:
    #     bin_url_all = bin_url + file_name.values + ".bin"

    # Set default parameters (see the simulate function in automate_plume_viz.py to get more details)
    emit_time_hrs = 1
    duration = (end_d[0] - start_d[0]).days * 24  + (end_d[0] - start_d[0]).seconds / 3600
    filter_ratio = 0.8

    # Run the simulation for each date in parallel (be aware of the memory usage)
    arg_list = []
    print("Running hysplit simulation with duration: %s hours" % duration)
    for i in range(len(bin_file_all)):
        arg_list.append((start_time_eastern_all[i], bin_file_all[i], sources,
            emit_time_hrs, duration, filter_ratio, use_forecast)) #bin_url_all[i]
    pool = Pool(num_workers)
    pool.starmap(simulate_worker, arg_list)
    pool.close()
    pool.join()


def remove_files(directory):
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".hrrra") or file.endswith(".bin"):
                    print("Removing file: %s" % os.path.join(root, file))
                    os.remove(os.path.join(root, file))

        cache_path = os.path.join(directory,"earthtime/air-data/dispersionStiltCache")  # Specify the directory path

        # Get the list of files in the directory
        source_folders = os.listdir(cache_path)

        # Iterate over the files and remove them one by one
        for folder in source_folders:
            folder_path = os.path.join(cache_path, folder)
            print("Removing folder: %s" % folder_path)
            shutil.rmtree(folder_path)

    except OSError:
        pass


def main(argv):
    if len(argv) < 2:
        print("Usage:")
        print("python main.py generate_earthtime_data")
        print("python main.py run_hysplit")
        print("python main.py remove_files")
        return

    program_start_time = time.time()

    #PATH = argv[2]

    # IMPORTANT: specify the path on the server that stores your particle bin files
    # bin_root = "[YOUR_PATH]/bin/"
    # hrr_root = "/home/nholzbach/hysplit_data/earthtime/air-data/hrrr/"
    hrr_root = PATH+"/hysplit_data/earthtime/air-data/hrrr/"
    bin_root = PATH+"/hysplit_data/bin/"

    # IMPORTANT: specify the URL for accessing the bin files
    # bin_url = "https://[YOUR_URL_ROOT]/bin/"
    bin_url = "https://home/nholzbach/hysplit_data/bin/"

    # IMPORTANT: specify a list to indicate the starting and ending dates to proces
    # You can use two supporing functions to generate the list, see below for examples
    # date_list = get_time_range_list(["2019-03-05", "2019-03-06"], duration=24, offset_hours=3)
    # date_list = get_start_end_time_list("2019-04-01", "2019-05-01", offset_hours=3)

    # THIS IS JUST ONE DAY FOR TESTING, edit later
    # date_list = get_time_range_list(["2017-06-17"], duration=24, offset_hours=0)
    # inputted date_list:
    date = argv[2]
    date_list = get_time_range_list([date], duration=24, offset_hours=0)
    
    # IMPORTANT: specify an unique string to prevent your data from mixing with others
    # IMPORTANT: do not set prefix to "plume_" which is used by the CREATE Lab's project
    prefix = "run_"

    # IMPORTANT: specify a list of pollution sources
    sources = [
        {
            "dispersion_source":DispersionSource(name='Irvin', ID =1,lat=40.328015, lon=-79.903551, minHeight=0, maxHeight=50, emit_file = 'data/Glassport_emission.csv'),
            # "color": [250, 255, 99],
            "filter_out": .58
        },
        {
            "dispersion_source":DispersionSource(name='ET',ID =2,lat=40.392967, lon=-79.855709, minHeight=0, maxHeight=50, emit_file = 'data/ET_emission.csv'),
            # "color": [99, 255, 206],
            "filter_out": .74
        },
        {
            "dispersion_source":DispersionSource(name='Clairton',ID =3,lat=40.305062, lon=-79.876692, minHeight=0, maxHeight=50, emit_file = 'data/Liberty_emission.csv'),
            # "color": [206, 92, 247],
            "filter_out": .10
        },
        {
            "dispersion_source":DispersionSource(name='Cheswick',ID =4,lat=40.538261, lon=-79.790391, minHeight=0, maxHeight=50, emit_file = 'data/Harrison_emission.csv'),
            # "color": [255, 119, 0],
            "filter_out": .81
        },
        {
            "dispersion_source":DispersionSource(name='McConway',ID =5,lat=40.479019, lon=-79.960299, minHeight=0, maxHeight=50, emit_file = 'data/Lawrenceville_emission.csv')
        }
        ]

    # IMPORTANT: specify the location of the map that you want to show, using lat, lng, and zoom
    # ...(lat means latitude, lng means longitude, zoom means the zoom level of the Google Map)
    lat, lng, zoom = "40.42532", "-79.91643", "9.233"

    # IMPORTANT: if you do not want to show smell reports, set add_smell to False
    add_smell = False

    # IMPORTANT: if you want the thumbnail server to re-render video frames (e.g., for experiments), increase redo
    redo = 0


    # Set the number of partitions of the URL for the thumbnail server to process in parallel
    url_partition = 4

    # Set the prefix of the names of the EarthTime layers
    # ...(will only affect the layers shown on the EarthTime system)
    name_prefix = "PARDUMP "

    # Set the credit of the EarthTime layer
    # ...(will only affect the layers shown on the EarthTime system)
    credits = "CREATE Lab"

    # Set the category of the EarthTime layer
    # ...(will only affect the layers shown on the EarthTime system)
    category = "Plume Viz"

    # Set the size of the output video (for both width and height)
    # img_size = 540

    #Optionally choose to use forecast meteorology instead of reanalysis files
    use_forecast = False


    #today = date.today().strftime("%Y-%m-%d")


    # Sanity checks
    assert(bin_root is not None), "you need to edit the path for storing hysplit particle files"
    # assert(video_root is not None), "you need to edit the path for storing video files"
    assert(bin_url is not None), "you need to edit the URL for accessing the particle files"
    # assert(video_url is not None), "you need to edit the URL for accessing the video files"
    assert(date_list is not None),"you need to specify the dates to process"
    assert(prefix is not None),"you need to specify the prefix of the unique share url"
    assert(len(sources) > 0),"you need to specify the pollution sources"
    assert(lat is not None),"you need to specify the latitude of the map"
    assert(lng is not None),"you need to specify the longitude of the map"
    assert(zoom is not None),"you need to specify the zoom level of the map"
    assert(argv[2] is not None),"you need to specify the DATE"
    

    # Run the following line first to generate EarthTime layers
    # IMPORTANT: you need to copy and paste the generated layers to the EarthTime layers CSV file
    # ...check the README file about how to do this
    if argv[1] in ["generate_earthtime_data", "run_hysplit"]:

        start_d, end_d, file_name = generate_earthtime_data(date_list,
                prefix, add_smell, lat, lng, zoom,
                credits, category, name_prefix)

    # Then run the following to create hysplit simulation files
    # IMPORTANT: after creating the bin files, you need to move them to the correct folder for public access
    # ... check the README file about how to copy and move the bin files
    # IMPORTANT: if you are doing experiments on creating the particle files,
    # ...make sure you set the input argument "bin_url" of the run_hysplit function to None
    # ...otherwise the code will not run because the particle files aleady exist in the remote URLs
    if argv[1] == "run_hysplit":
        run_hysplit(sources, bin_root, start_d, end_d, file_name, bin_url=bin_url, use_forecast=use_forecast)

    if argv[1] == "remove_files":
        directory = current_path
        remove_files(directory)


    program_run_time = (time.time()-program_start_time)/60
    print("Took %.2f minutes to run the program" % program_run_time)



if __name__ == "__main__":
    main(sys.argv)
