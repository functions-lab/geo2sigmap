import concurrent.futures
import concurrent
import multiprocessing

import osmnx as ox
import numpy as np
from tqdm import tqdm
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


min_lat = 34.898963
max_lat = 36.460291
min_lon = -80.728875
max_lon = -75.727737
##0.01 Decimal Degrees equals to 1.11 kilometre
step_size_lon = 0.01
step_size_lat = 0.01 * 0.657657
print(step_size_lon)

from pyproj import Transformer

to4326 = Transformer.from_crs("EPSG:6933", "EPSG:4326")
to6933 = Transformer.from_crs("EPSG:4326", "EPSG:6933")
print(to6933.transform(36.460291, -75.727737))

min_lat_6933 = 4188062.1742998925
max_lat_6933 = 4350591.699854682
min_lon_6933 = -7789228.857589593
max_lon_6933 = -7306687.654948185
print(to4326.transform(-7789228.857589593 + 1000, 4188062.1742998925 + 1000))

# ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"
# overpass_rate_limit = False


#ox.config(overpass_endpoint="https://overpass.kumi.systems/api/interpreter",overpass_rate_limit=False,use_cache=True, cache_only_mode=True)
ox.config(overpass_endpoint="http://tc319-srv1.egr.duke.edu:23412/api/interpreter",overpass_rate_limit=False,use_cache=False,cache_only_mode=False)

#ox.config(overpass_endpoint="http://tc319-srv1.egr.duke.edu:23412/api/interpreter",overpass_rate_limit=False,use_cache=False,cache_only_mode=False)
#ox.config(overpass_endpoint="http://24.163.79.230:12345/api",overpass_rate_limit=False,use_cache=False,cache_only_mode=False)
# geometries = ox.geometries.geometries_from_bbox(north=bbox[1], south=bbox[3],
#                                                 east=bbox[0], west=bbox[2], tags={'building': True})


# def isCity(bbox):
#     #print(bbox)
#     geometries = ox.geometries.geometries_from_bbox(bbox[1], south=bbox[3],
#                                                     east=bbox[0], west=bbox[2], tags={'building': True})
#     # Check the data type
#     #print(geometries)
#     #geometries.set_crs(epsg=4326, inplace=True)
#     # print(geometries)
#     # if not geometries.empty :
#     #     geometries.plot()
#     geometries = geometries.to_crs("EPSG:6933")
#     building_ratio = geometries.area.sum() / 998978
#     # print(building_ratio)
#     # if building_ratio > 0.1:
#     #     return 1
#     #
#     # else:
#     #     return 0

def compute_building_to_land_ration(tmp_top_left_lat,tmp_top_left_lon ):
    tmp_4326_bottom_left = to4326.transform(tmp_top_left_lon, tmp_top_left_lat)
    tmp_4326_top_right = to4326.transform(tmp_top_left_lon + 1000, tmp_top_left_lat + 1000)
    tmp_4326_min_lat = tmp_4326_bottom_left[0]
    tmp_4326_max_lat = tmp_4326_top_right[0]
    tmp_4326_min_lon = tmp_4326_bottom_left[1]
    tmp_4326_max_lon = tmp_4326_top_right[1]
    top_left = (tmp_4326_max_lat, tmp_4326_min_lon)
    top_right = (tmp_4326_max_lat, tmp_4326_max_lon)
    bottom_left = (tmp_4326_min_lat, tmp_4326_min_lon)
    bottom_right = (tmp_4326_min_lat, tmp_4326_max_lon)

    # print("Top Left: ", top_left)
    # print("Top Right:", top_right)
    # print("Bottom Left:", bottom_left)
    # print("Bottom Right:", bottom_right)

    # (-78.94514, 36.00578, -78.93646, 35.99939)
    bbox = (tmp_4326_min_lon, tmp_4326_max_lat, tmp_4326_max_lon, tmp_4326_min_lat)
    try:
        geometries = ox.geometries.geometries_from_bbox(bbox[1], south=bbox[3],
                                                        east=bbox[0], west=bbox[2], tags={'building': True})
    except:
        return bbox, 0
    geometries = geometries.to_crs("EPSG:6933")
    building_ratio = geometries.area.sum() / 998978
    return bbox, building_ratio


count = 0
cc = 0


with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
    with tqdm(total= 163 * 483) as pbar:


        for tmp_top_left_lat in np.arange(min_lat_6933, max_lat_6933, 1000):
            futures = []
            for tmp_top_left_lon in np.arange(min_lon_6933, max_lon_6933, 1000):
                a_result = executor.submit(compute_building_to_land_ration, tmp_top_left_lat,tmp_top_left_lon)
                futures.append(a_result)

                #bbox, building_ratio = compute_building_to_land_ration(tmp_top_left_lat, tmp_top_left_lon)
                #print(building_ratio)
            file1 = open("res3_srv1_33.txt", "a")
            for future in futures:
                bbox, building_ratio = future.result()
                file1.writelines('{},{}\n'.format(bbox, building_ratio))
                pbar.update(1)
            file1.close()


