from pyquadkey2 import quadkey
import ogr2osm
import geopy.distance

# import xml.etree.ElementTree as ET
from lxml import etree

import os
from pprint import pprint
import numpy as np

import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

lat, lon = 36.003045222704806, -78.93705434567853
lat_lon_height_d = {}
curr_gdf = {}
# print(str(quadkey.from_geo((lat, lon), 9)))


def add_height(path_to_osm):
    tree = etree.parse(path_to_osm).getroot()
    minlat, minlon, maxlat, maxlon = (None, None, None, None)
    for e in tree:
        if e.tag == 'bounds':
            minlat, minlon, maxlat, maxlon = float(e.get('minlat')), float(e.get('minlon')), float(e.get('maxlat')), float(e.get('maxlon'))
            curr_QuadKey = str(quadkey.from_geo((round((minlat + maxlat) / 2, 6), round((minlon + maxlon) / 2, 6)), 9))
            global curr_gdf
            if curr_QuadKey not in curr_gdf.keys():
                # add_geodataframe(round((minlat + maxlat) / 2, 6), round((minlon + maxlon) / 2, 6))
                add_geodataframe(round(lat, 6), round(lon, 6))
            break
    node_coord_d = {}
    for node in tree.findall('node'):
        node_coord_d[node.get('id')] = [float(node.get('lat')), float(node.get('lon'))]
    ht_none_counter = 0
    to_del_list = ['building:levels', 'height']
    for e in tree.findall('./*/*'):
        if e.get('k') == "building":
            lat_lon_list = []  # this is list of coords of current node (most likely tag "way")
            for f in e.getparent().findall('nd'):
                print(f.get('ref'))
                lat_lon_list.append(node_coord_d[f.get('ref')])
            print(lat_lon_list)
            lat_lon_list = np.array(lat_lon_list)
            print(lat_lon_list)
            if max(lat_lon_list[:, 0]) - min(lat_lon_list[:, 0]) > 1e-6 or max(lat_lon_list[:, 1]) - min(lat_lon_list[:, 1]) > 1e-6:
                print('loc diff in lat_lon_list too large')
            lat_lon_cent = [np.average(lat_lon_list[:, 0]), np.average(lat_lon_list[:, 1])]
            ht = search_height(*lat_lon_cent)

            if ht is not None:
                e_parent = e.getparent()
                for to_del in to_del_list:
                    to_delete_element = e_parent.find(f'.//tag@[k="{to_del}"]')
                    if to_delete_element is not None:
                        e_parent.remove(to_delete_element)
                etree.SubElement(e.getparent(), "tag", k="height", v=str(ht))
                etree.SubElement(e.getparent(), "tag", k="building:levels", v=str(int(round(ht / 4.3))))
            else:
                print('height is None ' + ht_none_counter, 'meaning lat, lon pair isnt there')
                ht_none_counter += 1
    tree.write(os.path.join(os.path.dirname(path_to_osm), os.path.basename(path_to_osm).split('.')[0] + "_new" + '.osm'), 
               pretty_print=True, xml_declaration=True, encoding="utf-8")

    return


def search_height(lati, long):
    global lat_lon_height_d
    for lat_n, long_n in lat_lon_height_d.keys():
        if geopy.distance.distance((lati, long), (lat_n, long_n)).meters < 4:
            return lat_lon_height_d[(lat_n, long_n)]
    print('missed')
    return None

    
def add_geodataframe(lati, long):
    QuadKey = str(quadkey.from_geo((lati, long), 9))
    dataset_links = pd.read_csv("https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv")
    desired_row = dataset_links[dataset_links.QuadKey == int(QuadKey)]

    df = pd.read_json(desired_row.iloc[-1]['Url'], lines=True)
    df['geometry'] = df['geometry'].apply(shape)
    gdf = gpd.GeoDataFrame(df, crs=4326)
    global lat_lon_height_d, curr_gdf
    for _, row in gdf.iterrows():
        # inserts key=(lat, lon), val=height
        lat_lon_height_d[(row['geometry'].centroid.y, row['geometry'].centroid.x)] = row['properties']['height']
    curr_gdf[lati, long] = gdf
    return 


def convert_to_osm(in_fname, out_fname):
    translation_object = ogr2osm.TranslationBase()
    datasource = ogr2osm.OgrDatasource(translation_object)
    datasource.open_datasource(in_fname)
    osmdata = ogr2osm.OsmData(translation_object)
    osmdata.process(datasource)
    datawriter = ogr2osm.OsmDataWriter(out_fname)
    osmdata.output(datawriter)
    return


# add_geodataframe(lat, lon)
# convert_to_geojson(lat, lon)
add_height(path_to_osm='osm/pci_EF.osm')
# convert_to_osm(convert_to_geojson(lat, lon), 'duke.osm')
