"""
Automate the plume simulation using hysplit model
"""

import os, re, datetime, json, pytz, subprocess, time, shutil, requests, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.parse
import urllib.request
from multiprocessing.dummy import Pool
from os import listdir
from os.path import isfile, join, isdir
from zipfile import ZipFile
from utils import subprocess_check
from pardump_util import findInFolder, create_multisource_bin
from cached_hysplit_run_lib import getMultiHourDispersionRunsParallel, parse_eastern, HysplitModelSettings, InitdModelType, CachedDispersionRun


hysplit_root = "/home/nholzbach/hysplit/"

def exec_ipynb(filename_or_url):
    """Load other ipython notebooks and import their functions"""
    nb = (requests.get(filename_or_url).json() if re.match(r'https?:', filename_or_url) else json.load(open(filename_or_url)))
    if(nb['nbformat'] >= 4):
        src = [''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code']
    else:
        src = [''.join(cell['input']) for cell in nb['worksheets'][0]['cells'] if cell['cell_type'] == 'code']
    tmpname = '/tmp/%s-%s-%d.py' % (os.path.basename(filename_or_url),
                                    datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'),
                                    os.getpid())
    src = '\n\n\n'.join(src)
    open(tmpname, 'w').write(src)
    code = compile(src, tmpname, 'exec')
    exec(code, globals())

def get_start_end_time_list(start_date_eastern, end_date_eastern, duration=24, offset_hours=3):
    """
    Given starting and ending date string, get a list of starting and ending datetime objects

    Input:
        start_date_eastern: the date to start in EST time, e.g., "2019-01-01"
        end_date_eastern: the date to end in EST time, e.g., "2020-01-01"
        offset_hour: time offset in hours, for example, if this is 3, then it starts from 12-3=9 p.m. instead of 12 a.m.

    Output:
        start_d: a pandas DatetimeIndex object, indicating the list of starting times
        end_d: a pandas DatetimeIndex object, indicating the list of ending times
    """
    offset_d = pd.Timedelta(offset_hours, unit="h")
    start_d = pd.date_range(start=start_date_eastern, end=end_date_eastern, closed="left", tz="US/Eastern") - offset_d
    end_d = start_d + pd.Timedelta(duration, unit="h")
    return (start_d, end_d)

def get_time_range_list(start_date_str_list, duration=24, offset_hours=3):
    """
    Convert lists of starting and ending date strings to objects

    Input:
        start_date_str_list: a list of date strings, e.g., ["2019-04-23", "2019-12-22", "2020-02-05"]
        duration: the number of hours for each time range, e.g., 24
        offset_hour: time offset in hours, for example, if this is 3, then it starts from 12-3=9 p.m. instead of 12 a.m.

    Output:
        start_d: a pandas DatetimeIndex object, indicating the list of starting times
        end_d: a pandas DatetimeIndex object, indicating the list of ending times
    """
    offset_d = pd.Timedelta(offset_hours, unit="h")
    start_d = pd.DatetimeIndex(data=start_date_str_list, tz="US/Eastern") - offset_d
    end_d = start_d + pd.Timedelta(duration, unit="h")
    return (start_d, end_d)

def generate_metadata(start_d, end_d, video_start_delay_hrs=0, url_partition=4, redo=0,
        prefix="banana_", add_smell=True, lat="40.42532", lng="-79.91643", zoom="9.233", credits="CREATE Lab",
        category="Plume Viz", name_prefix="PARDUMP ", file_path="https://cocalc-www.createlab.org/test/"):
    """
    Generate the EarthTime layers and the thumbnail server urls that can be called later to obtain video frames

    Input:
        start_d: a pandas DatetimeIndex object, indicating the list of starting times
        end_d: a pandas DatetimeIndex object, indicating the list of ending times
        video_start_delay_hrs: number of hours that the model should run before the video starts
        url_partition: the number of partitions for the thumbnail server request for getting images of video frames
        img_size: the size of the output video (e.g, 540 means 540px for both width and height)
        redo: this is a number to force the server to avoid using the cached file
        prefix: a string prefix for the generated unique share url identifier in the EarthTime layers
        add_smell: a flag to control if you want to add the smell reports to the visualization
        lat: a string that indicates the latitude of the EarthTime base map
        lng: a string that indicates the longitude of the EarthTime base map
        zoom: a string that indicates the zoom level of the EarthTime base map
        credits: a string to fill out the "Credits" column in the output EarthTime layers file
        category: a string to fill out the "Category" column in the output EarthTime layers file
        name_prefix: a string predix for the "Name" column in the output EarthTime layers file
        file_path: an URL path to indicate the location of your hysplit bin files

    Output:
        df_layer: the pandas dataframe for the EarthTime layer document
        df_share_url: the pandas dataframe for the share urls
        df_img_url: the pandas dataframe for the thumbnail server urls to get images of video frames
        file_name: a list of file names that are used for saving the hysplit bin files
    """
    if url_partition < 1:
        url_partition = 1
        print("Error! url_partition is less than 1. Set the url_partition to 1 to fix the error.")

    # Create rows in the EarthTime layer document
    df_template = pd.read_csv("data/earth_time_layer_template.csv")
    df_layer = pd.concat([df_template]*len(start_d), ignore_index=True)
    vid_start_d = start_d + pd.DateOffset(hours=video_start_delay_hrs)
    start_d_utc = vid_start_d.tz_convert("UTC")
    end_d_utc = end_d.tz_convert("UTC")
    file_name = prefix + start_d_utc.strftime("%Y%m%d%H%M") + "_" + end_d_utc.strftime("%Y%m%d%H%M")
    df_layer["Start date"] = start_d_utc.strftime("%Y%m%d%H%M%S")
    df_layer["End date"] = end_d_utc.strftime("%Y%m%d%H%M%S")
    df_layer["Share link identifier"] = file_name
    df_layer["Name"] = name_prefix + start_d_utc.strftime("%Y-%m-%d")
    df_layer["URL"] = file_path + file_name + ".bin"
    df_layer["Category"] = category
    df_layer["Credits"] = credits

    # Create rows of share URLs
    # et_root_url = "https://headless.earthtime.org/#"
    # et_part = "v=%s,%s,%s,latLng&ps=2400&startDwell=0&endDwell=0" % (lat, lng, zoom)
    # ts_root_url = "https://thumbnails-earthtime.cmucreatelab.org/thumbnail?"
    # ts_part = "&width=%d&height=%d&format=zip&fps=30&tileFormat=mp4&startDwell=0&endDwell=0&fromScreenshot&disableUI&redo=%d" % (img_size, img_size, redo)
    # share_url_ls = [] # EarthTime share urls
    # dt_share_url_ls = [] # the date of the share urls
    # img_url_ls = [] # thumbnail server urls
    # dt_img_url_ls = [] # the date of the thumbnail server urls

    # NOTE: this part is for testing the new features that override the previous ones
    df_layer["Vertex Shader"] = "WebGLVectorTile2.particleAltFadeVertexShader"
    df_layer["Fragment Shader"] = "WebGLVectorTile2.particleAltFadeFragmentShader"
    et_root_url = "https://headless.earthtime.org/#"

    for i in range(len(start_d_utc)):
        sdt = start_d_utc[i]
        edt = end_d_utc[i]
        # Add the original url
        sdt_str = sdt.strftime("%Y%m%d%H%M%S")
        edt_str = edt.strftime("%Y%m%d%H%M%S")
        #date_str = sdt_str[:8]
        date_str = sdt_str[:12] + "_" + edt_str[:12]
        bt = "bt=" + sdt_str + "&"
        et = "et=" + edt_str + "&"
        if add_smell:
            #TODO: create a more stable smell report layer
            l = "l=mb_labeled,smell_my_city_pgh_reports_top2," + file_name[i] + "&"
        else:
            l = "l=mb_labeled," + file_name[i] + "&"
        # share_url_ls.append(et_root_url + l + bt + et + et_part)
        # dt_share_url_ls.append(date_str)
        # Add the thumbnail server url
        time_span = (edt - sdt) / url_partition
        for j in range(url_partition):
            std_j = sdt + time_span*j
            edt_j = std_j + time_span
            std_j_str = std_j.strftime("%Y%m%d%H%M%S")
            edt_j_str = edt_j.strftime("%Y%m%d%H%M%S")
            bt_j = "bt=" + std_j_str + "&"
            et_j = "et=" + edt_j_str + "&"
            rt = "root=" + urllib.parse.quote(et_root_url + l + bt_j + et_j + et_part, safe="") + "&"
            img_url_ls.append(ts_root_url + rt + ts_part)
            dt_img_url_ls.append(date_str)
    df_share_url = pd.DataFrame(data={"share_url": share_url_ls, "date": dt_share_url_ls})
    df_img_url = pd.DataFrame(data={"img_url": img_url_ls, "date": dt_img_url_ls})

    # return the data
    return (df_layer, df_share_url, df_img_url, file_name)


def simulate(start_time_eastern, o_file, sources, emit_time_hrs=0, duration=24, filter_ratio=0.8,
        useForecast=False):
    """
    Run the HYSPLIT simulation

    Input:
        start_time_eastern: for different dates, use format "2020-03-30 00:00"
        o_file: file path to save the simulation result, e.g., "/projects/cocalc-www.createlab.org/pardumps/test.bin"
        sources: dict containing location of the sources of pollution (DispersionSource objects), color, and ratio of points to filter
        emit_time_hrs: affects the emission time for running each Hysplit model
        duration: total time (in hours) for the simulation, use 24 for a total day, use 12 for testing
        filter_ratio: the ratio that the points will be dropped (e.g., 0.8 means dropping 80% of the points)
        hysplit_root: the root directory of the hysplit software
    """
    print("="*100)
    print("="*100)
    print("start_time_eastern: %s" % start_time_eastern)
    print("o_file: %s" % o_file)

    # Check and make sure that the o_file path is created
    check_and_create_dir(o_file)

    # Run simulation and get the folder list (the generated files are cached)
    path_list = []
    for source in sources:
        path_list += getMultiHourDispersionRunsParallel(
                source["dispersion_source"],
                parse_eastern(start_time_eastern),
                emit_time_hrs,
                duration,
                HysplitModelSettings(initdModelType=InitdModelType.ParticleHV, hourlyPardump=False),
                useForecast=useForecast)
    print("Simulation done, len(path_list)=%d" % len(path_list))


    # Check and make sure that the o_file path is created
    cdump_txt_path_list = []
    # description = pd.DataFrame([[f"Model type:ParticleHV, Day:{start_time_eastern}, Duration:{duration}, emission_time_hrs:{emit_time_hrs}"]])
    cdump_df = pd.DataFrame()
    # cdump_df = pd.concat([description, cdump_df], axis=0)
    for folder in path_list:
        if not findInFolder(folder,"cdump.txt"):
            cdump = findInFolder(folder, "cdump")
            cmd = hysplit_root + "exec/con2asc -i%s -s -t -v -x" % cdump
            if cdump.find('.txt') == -1:
                cdump_txt_path_list.append(cdump+".txt")
                cdump_txt = cdump+".txt"
            subprocess_check(cmd)
            print("con2asc done for %s" % cdump)
        else:
            print("cdump.txt already exists")
            cdump_txt = findInFolder(folder,'cdump.txt')
            cdump_txt_path_list.append(cdump_txt)


        # extract unique info for the run
        coord = (folder.split("/")[7].split("_")[0])
        # connect this to source id?? or just add it to the df
        datetime = folder.split("/")[-1]

        # make a dataframe of txt file
        df = pd.read_fwf(cdump_txt)
        df.columns = df.columns.str.replace(r'^TEST.*', 'TEST', regex=True)
        # add coord as another column
        df = df.assign(source=coord)
        # df = df.insert(0, 'source', coord, allow_duplicates=False)
        # add df to the big df
        cdump_df = pd.concat([cdump_df, df], axis=0, join='outer')

    # Write the description and data to a CSV file
    # this is a bit hardcoded, need to be able to change the emission scheme automatically
    description = f"Model type:ParticleHV,particle dispersion scheme: 1, Day:{start_time_eastern}, Duration:{duration}hrs, emission_time_hrs:{emit_time_hrs}hrs, emission scheme: {sources}  \n \n"
    run_info = "run_%s_%shr" % (start_time_eastern, duration)
    with open('/home/nholzbach/hysplit_data/results/%s.csv' % run_info, 'w') as f:
        f.write(description)
        cdump_df.to_csv(f, header=True, index=False)


    print("cdumps all collected in home/nholzbach/hysplit_data/results/%s.csv" % run_info )




def is_url_valid(url):
    """Check if the url is valid (has something)"""
    try:
        r = requests.head(url)
        return r.status_code == requests.codes.ok
    except Exception:
        traceback.print_exc()
        return False


def simulate_worker(start_time_eastern, o_file, sources, emit_time_hrs, duration, filter_ratio, useForecast=False):
    """
    The parallel worker for hysplit simulation

    Input:
        o_url: if not None, check if the URL for the particle file already exists in the remote server
        (for other input parameters, see the docstring of the simulate function)
    """
    # Skip if the file exists in local
    if os.path.isfile(o_file):
        print("File exists in local %s" % o_file)
        return True


    # Perform HYSPLIT model simulation
    try:
        simulate(start_time_eastern, o_file, sources,
                emit_time_hrs=emit_time_hrs, duration=duration, filter_ratio=filter_ratio, useForecast=useForecast)
        return True
    except Exception:
        print("-"*60)
        print("Error when creating %s" % o_file)
        traceback.print_exc()
        print("-"*60)
        return False



def check_and_create_dir(path):
    """Check if a directory exists, if not, create it"""
    if path is None: return
    dir_name = os.path.dirname(path)
    if dir_name != "" and not os.path.exists(dir_name):
        try: # this is used to prevent race conditions during parallel computing
            os.makedirs(dir_name)
            os.chmod(dir_name, 0o777)
        except Exception:
            traceback.print_exc()


def del_dir(dir_p):
    """Delete a directory and all its contents"""
    if not os.path.isdir(dir_p): return
    try:
        shutil.rmtree(dir_p)
    except Exception:
        traceback.print_exc()


def get_all_file_names_in_folder(path):
    """Return a list of all files in a folder"""
    return [f for f in listdir(path) if isfile(join(path, f))]


def get_all_dir_names_in_folder(path):
    """Return a list of all directories in a folder"""
    return [f for f in listdir(path) if isdir(join(path, f))]
