import subprocess
import concurrent
from concurrent.futures import wait
import os
from os.path import join, dirname
# import uuid
import argparse
import logging
import requests
# from dotenv import load_dotenv

# Create a logger
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)  # Set the logging level

console_handler = logging.StreamHandler()  # By default, this directs logs to stdout
console_handler.setLevel(logging.INFO)  # Set the handler's logging level

# Create a formatter and set it on the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# # Add the file handler to the logger
logger.addHandler(console_handler)


parser = argparse.ArgumentParser(description='Process latitude and longitude ranges.')

# Adding arguments
parser.add_argument('--blender-path', default="~/blender-git/build_linux_release/bin/blender", type=str, help='Blender binary file path.')
parser.add_argument('--command-line-script-path', default="blender_test_command_line.py", type=str, help='Command line python script path.')
parser.add_argument('--data-dir', type=str, help='Data folder', required=True)
parser.add_argument('--start-idx', default=0, type=int, help='The index of area which you prefer to start the process.')
parser.add_argument('--end-idx', type=int, default=80,help='The index of area which you prefer to stop the process. -1 means no stop.')
# parser.add_argument('--base-path', type=str, default='data/generated', help='Base path to store the generated data.')
parser.add_argument('--max-process', type=int, default=5, help='Maximum process used for generate data.')
# parser.add_argument('--max-thread', type=int, default=5, help='Maximum thread used for generate data.')
parser.add_argument('--land-type', choices=['terrain', 'plane'], default='plane', help='The type of land when rendering the 3D meshs. Note this is a experimental feature, Sionna still not supprt dynamical altitude level ray tracing.')
parser.add_argument('--res-file', type=str, default="Filtered_Area_b2l_result.txt", help='Area dimension.')

# Parse the arguments
args = parser.parse_args()
logger.info("All settings used:")
for k,v in sorted(vars(args).items()):
    logger.info("{0}: {1}".format(k,v))




# load_dotenv(join(dirname(__file__), '../.env'))

PROJECT_BASE_PATH = os.path.join(dirname(__file__),"../")

BASE_PATH = args.data_dir

# BLENDER_PATH should be the path I built, since the things are enabled
#BLENDER_PATH = os.path.join(PROJECT_BASE_PATH,os.environ.get('BLENDER_PATH'))
BLENDER_PATH = os.path.expanduser(args.blender_path)

if not os.path.exists(BLENDER_PATH):
    logger.error("Blender path is not exist!")
    exit()



BLENDER_COMMAND_LINE_PATH = os.path.join(dirname(__file__), args.command_line_script_path)
BLENDER_OSM_DOWNLOAD_PATH = os.path.join(args.data_dir, "OSM_download")

START_FROM_IDX = args.start_idx
STOP_AT_IDX = args.end_idx
NUM_OF_PROCESS = args.max_process
DECIMATE_FACTOR = 1
TERRAIN_OR_PLANE = args.land_type
RES_FILE_NAME = args.res_file


def install_package(package_name):
        print("Start pip install process")
        python_exe = os.path.join(os.path.dirname(BLENDER_PATH),"3.3/python/bin/python3.10")
        file_path  = os.path.join(os.path.dirname(python_exe), "get-pip.py")
        try:

            response = requests.get("https://bootstrap.pypa.io/get-pip.py")
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                print(f"File downloaded successfully to {file_path}")
            else:
                print("Failed to download the file")
                exit()
            # path to python.exe
            
            # upgrade pip
            #\subprocess.call([python_exe, "-m", "ensurepip"])
            subprocess.call([python_exe, file_path])
            subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

            # install required packages
            subprocess.call([python_exe, "-m", "pip", "install", package_name])

            print("DONE")
            return
        except Exception as e:
            print("got error in install pip")
            raise e





def check_display():

    display = os.environ.get('DISPLAY')

    if display:
        return True
    else:
        return False




MITSUBA_EXPORT_BUILDINGS = 'y'  # controls whether mitsuba will export the XML file
XML_TO_BUILDING_MAP = 'n'  # controls whether to use existing XML to produce building map

def splitting_a_line(lll, uuid_incl='n'):
    lll = lll.replace('(', '')
    lll = lll.replace(')', '')
    lll = lll.replace('\n', '')
    lll = lll.split(',')
    # file format: (minLon,maxLat,maxLon,minLat),percent,idx_uuid
    if uuid_incl == 'y':
        minLon, maxLat, maxLon, minLat, perc, idx_uuid = [k for k in lll]
        return float(minLon), float(maxLat), float(maxLon), float(minLat), float(perc), idx_uuid
    else:
        minLon, maxLat, maxLon, minLat, perc = [float(k) for k in lll]
        return minLon, maxLat, maxLon, minLat, perc


if __name__ == '__main__':
    install_package("mitsuba==3.0.1")
    precheck_result = subprocess.run([
                                        BLENDER_PATH,
                                        "--background",
                                        "--python",
                                        BLENDER_COMMAND_LINE_PATH, "--",
                                        "--precheck",
                                        '--peoject_base_path', str(PROJECT_BASE_PATH) ], 
                                    capture_output=True, text=True)
    output = precheck_result.stdout
    error = precheck_result.stderr

    # Print output and error messages
    print("Output:", output)
    print("Error:", error)
    print(precheck_result.returncode)
    if precheck_result.returncode != 0:
        # Get the output and error messages (if needed)
        logger.error("Fatal error when installing the Blender addons!")
        exit(-1)

    print(BASE_PATH)
    print("start")
    os.makedirs(os.path.join(BASE_PATH , 'height_at_origin/'), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH , 'Bl_terrain_npy/'), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH , 'Bl_building_npy/'), exist_ok=True)
    os.makedirs(os.path.join(BASE_PATH , 'Bl_xml_files/'), exist_ok=True)
    idx_uuid_xml = [f for f in os.listdir(os.path.join(BASE_PATH , 'Bl_xml_files/'))
                    if os.path.isdir(os.path.join(BASE_PATH , 'Bl_xml_files/',f)) and f != '.DS_Store']
    with open(os.path.join(BASE_PATH , RES_FILE_NAME), 'r') as loc_fPtr:
        lines = loc_fPtr.readlines()
    futures = []



    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_OF_PROCESS) as executor:
        for idx, line in enumerate(lines):
            minLonOut, maxLatOut, maxLonOut, minLatOut, percent, idx_uuid = splitting_a_line(lll=line, uuid_incl='y')
            if idx < START_FROM_IDX:
                continue
            if idx >= STOP_AT_IDX != -1:
                break
            if XML_TO_BUILDING_MAP == 'y':
                #import_path = BASE_PATH + 'Bl_xml_files/' + idx_uuid + '/' + idx_uuid + '.xml'
                import_path = os.path.join(BASE_PATH , 'Bl_xml_files',idx_uuid + '/' + idx_uuid + '.xml')
                if not os.path.exists(import_path):
                    continue  # skip idx_uuids that didn't get exported to XML files in previous runs
            # file format: (minLon,maxLat,maxLon,minLat),percent,idx_uuid\n
            # print(' '.join([BLENDER_PATH,  # "--background",
            #                 "--python",
            #                 BLENDER_COMMAND_LINE_PATH.replace(' ', '\ '), "--",
            #                 "--idx", str(idx),
            #                 "--minLon", str(minLonOut),
            #                 "--maxLat", str(maxLatOut),
            #                 "--maxLon", str(maxLonOut),
            #                 "--minLat", str(minLatOut),
            #                 "--building_to_area_ratio", str(percent),
            #                 "--decimate_factor", str(DECIMATE_FACTOR),
            #                 "--BASE_PATH", str(BASE_PATH).replace(' ', '\ '),
            #                 "--BLENDER_OSM_DOWNLOAD_PATH", str(BLENDER_OSM_DOWNLOAD_PATH).replace(' ', '\ '),
            #                 "--idx_uuid", str(idx_uuid),
            #                 '--terrain_or_plane', TERRAIN_OR_PLANE,
            #                 '--export_buildings', MITSUBA_EXPORT_BUILDINGS,
            command_list = [
                     BLENDER_PATH, "--background",
                     "--python",
                     BLENDER_COMMAND_LINE_PATH, "--",
                     "--idx", str(idx),
                     "--minLon", str(minLonOut),
                     "--maxLat", str(maxLatOut),
                     "--maxLon", str(maxLonOut),
                     "--minLat", str(minLatOut),
                     "--building_to_area_ratio", str(percent),
                     "--decimate_factor", str(DECIMATE_FACTOR),
                     "--BASE_PATH", str(BASE_PATH),
                     "--BLENDER_OSM_DOWNLOAD_PATH", str(BLENDER_OSM_DOWNLOAD_PATH),
                     "--idx_uuid", str(idx_uuid),
                     '--terrain_or_plane', TERRAIN_OR_PLANE,
                     '--export_buildings', MITSUBA_EXPORT_BUILDINGS,
                     '--xml_to_building_map', XML_TO_BUILDING_MAP,
                     '--peoject_base_path', str(PROJECT_BASE_PATH),
                     "--addons blosm"
                 ]
            if check_display:
                command_list = ["xvfb-run","-a"] + command_list
            #                 '--xml_to_building_map', XML_TO_BUILDING_MAP]))
            futures.append(executor.submit(
                subprocess.run,command_list,
                capture_output=True, text=True))
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                data = future.result()
                print('\n\n\n\n\n' + str(idx) + '\n' +str(data).replace("\\n","\n") + '\n\n\n\n\n')
            except Exception as e:
                print(e)
        wait(futures)