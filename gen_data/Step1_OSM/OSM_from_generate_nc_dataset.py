import concurrent
import multiprocessing
import asyncio
import os
from concurrent.futures import wait
from multiprocessing import Process
from threading import Thread
from time import sleep
import uuid
import osmnx as ox
import numpy as np
from tqdm.auto import tqdm

from dotenv import load_dotenv

import argparse
# 36.460291, -80.728875
#
#
#
# Union County
# North Carolina
# 34.898963, -80.694349
#
# 34.792208, -75.727737
#
# 36.426478, -75.916941


# 49.001474, -125.234374
# 24.709461, -124.032255
# 24.318717, -58.969696
# 49.695685, -65.554685


## For NC
# min_lat = 34.898963
# max_lat = 36.460291
# min_lon = -80.728875
# max_lon = -75.727737
##0.01 Decimal Degrees equals to 1.11 kilometre
# step_size_lon = 0.01
# step_size_lat = 0.01 * 0.657657
# print(step_size_lon)

N_Size = 512

from pyproj import Transformer


def compute_building_to_land_ration(tmp_top_left_lat, tmp_top_left_lon, queue,to4326):

    tmp_4326_bottom_left = to4326.transform(tmp_top_left_lon, tmp_top_left_lat)
    tmp_4326_top_right = to4326.transform(tmp_top_left_lon + 1000, tmp_top_left_lat + 1000)

    # print("Bottom Right:", bottom_right)

    # (-78.94514, 36.00578, -78.93646, 35.99939)
    bbox = (tmp_4326_bottom_left[1], tmp_4326_top_right[0], tmp_4326_top_right[1], tmp_4326_bottom_left[0])
    ox.settings.overpass_rate_limit = False
    ox.settings.use_cache = False
    ox.settings.cache_only_mode = False
    ox.settings.overpass_endpoint = os.environ.get('OSM_SERVER_ADDRESS') + "/api/interpreter"
    print(ox.settings.overpass_endpoint)
    try:

        geometries = ox.geometries.geometries_from_bbox(bbox[1], south=bbox[3],
                                                        east=bbox[0], west=bbox[2], tags={'building': True})
    except:
        queue.put((bbox, 0))
        return
    geometries = geometries.to_crs("EPSG:6933")
    building_ratio = geometries.area.sum() / N_Size / N_Size
    queue.put((bbox, building_ratio))

def producer(batch, queue):

    #ox.settings.overpass_endpoint = "http://tc319-srv1.egr.duke.edu:23412/api/interpreter"
    #ox.settings.overpass_endpoint = "http://10.237.197.245/api/interpreter"
    #ox.settings.overpass_endpoint = "http://192.168.1.164:8088/api"

    to4326 = Transformer.from_crs("EPSG:6933", "EPSG:4326")
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        for job in batch:
            executor.submit(compute_building_to_land_ration, job[0], job[1], queue, to4326)
        # executor.submit(compute_building_to_land_ration, job[0], job[1], queue, to4326) for job in batch
        # wait for tasks to cwait(futures)omplete
        #_ = wait(futures)




def consumer(queue, tqdm_size):

    pabar2 = tqdm(total=tqdm_size, position=0, desc="Saving", leave=True)
    # file1 = open("res3_srv1_whole_us.txt", "a")
    res = []
    while True:
        # get a unit of work

        item = queue.get(block=True)


        # check for stop
        if item is None:
            break
        # report
        bbox = item[0]
        building_ratio = item[1]

        res += ['(%f,%f,%f,%f),%f,%s\n' % (item[0][0], item[0][1], item[0][2], item[0][3], item[1],str(uuid.uuid4()))]

        #res += '{},{}\n'.format(bbox, building_ratio)
        if len(res) == 100:
            file1 = open(RES_FILE_PATH, "a")
            file1.writelines(res)
            res.clear()
            file1.close()

        pabar2.update(1)
    # file1.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process latitude and longitude ranges.')

    # Adding arguments
    parser.add_argument('--minilat', default=35.54728625208108, type=float, help='Minimum latitude')
    parser.add_argument('--maxlat', default=36.09139593781888, type=float, help='Maximum latitude')
    parser.add_argument('--minilon', default=-79.04640192793481, type=float, help='Minimum longitude')
    parser.add_argument('--maxlon', default=-78.42138748567791, type=float, help='Maximum longitude')

    # Parse the arguments
    args = parser.parse_args()

    # Accessing the values
    min_lat = args.minilat
    max_lat = args.maxlat
    min_lon = args.minilon
    max_lon = args.maxlon

    # Your logic here using these values
    print(f"Latitude range: {min_lat} to {max_lat}")
    print(f"Longitude range: {min_lon} to {max_lon}")

    load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    BASE_PATH = os.environ.get('BASE_PATH')
    RES_FILE_NAME = os.environ.get('RES_FILE_NAME')
    os.makedirs(BASE_PATH,exist_ok=True)
    RES_FILE_PATH = os.path.join(BASE_PATH,RES_FILE_NAME)


    # min_lat = 24.318717
    # max_lat = 49.695685
    # min_lon = -98.969696
    # max_lon = -69.953232


    # min_lat = 35.54728625208108
    # max_lat = 36.09139593781888
    # min_lon = -79.04640192793481
    # max_lon = -78.42138748567791



# res3_srv1_whole_us.txt
    # min_lat = 24.318717
    # max_lat = 49.695685
    # min_lon = -125.234374
    # max_lon = -98.969696

    # min_lat = 34.898963
    # max_lat = 36.460291
    # min_lon = -80.728875
    # max_lon = -75.727737

    to4326 = Transformer.from_crs("EPSG:6933", "EPSG:4326")
    to6933 = Transformer.from_crs("EPSG:4326", "EPSG:6933")
    print("res")
    print(to6933.transform(min_lat, min_lon))
    print(to6933.transform(max_lat, max_lon))

    min_lat_6933 = to6933.transform(min_lat, min_lon)[1]
    max_lat_6933 = to6933.transform(max_lat, max_lon)[1]
    min_lon_6933 = to6933.transform(min_lat, min_lon)[0]
    max_lon_6933 = to6933.transform(max_lat, max_lon)[0]
    print("min_lat:%f, max_lat:%f, min_lon:%f, max_lon:%f" % (min_lat_6933, max_lat_6933, min_lon_6933, max_lon_6933))
    print(to4326.transform(-7789228.857589593 + 1000, 4188062.1742998925 + 1000))

    #ox.settings.overpass_endpoint = "http://10.237.197.245/api/interpreter"



    m = multiprocessing.Manager()
    queue = m.Queue()
    job_queue = []
    for tmp_top_left_lat in np.arange(min_lat_6933, max_lat_6933, N_Size):
        for tmp_top_left_lon in np.arange(min_lon_6933, max_lon_6933, N_Size):
            job_queue.append((tmp_top_left_lat, tmp_top_left_lon))
    futures = []
    # pbar = tqdm(total=len(job_queue), position=0, leave=True)
    # pbar.update(1)
    consumer_process = Thread(target=consumer, args=(queue, len(job_queue)))
    consumer_process.start()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        batch_size = 10000
        for i in range(0, len(job_queue), batch_size):
            batch = job_queue[i:i + batch_size]  # the result might be shorter than batchsize at the end

            # do stuff with batch
        #for job in job_queue:
            a_result = executor.submit(producer, batch,queue)
            #futures.append(a_result)
            # pbar.update(1)
        executor.shutdown()
    queue.put(None)
    consumer_process.join()
