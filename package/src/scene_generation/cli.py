"""
CLI entry point for running the scene generation functionality.

Users can define the scene location by a rectangle in two ways:
1) Specifying four GPS corners directly, or
2) Providing one reference point plus width/height in meters and a corner/center position.

scenegenerationpipe \
    --data-dir test123 \
    --bbox -71.06025695800783 42.35128145107633 -71.04841232299806 42.35917815419112

TODO: Allow customized materials.
"""

import logging

from argparse import ArgumentParser
from .core import Scene
from .utils import (
    rect_from_point_and_size,
    print_if_int,
    get_package_version,
    PACKAGE_NAME,
)
from .itu_materials import ITU_MATERIALS

import xml.etree.ElementTree as ET
import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter


def setup_logging(log_file="debug.log"):
    """
    Configure logging such that:
      - All messages at DEBUG level or higher go to a file (log_file).
      - All messages at INFO level or higher go to the console (stdout).
      - The user can override log levels if desired.
    """
    # Create a base logger (the root logger)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # The root logger level should allow all messages

    # Create a formatter for consistent output
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    )
    # For console output, a simpler format:
    # e.g. "[INFO] Checking bounding box: http://bboxfinder.com/..."
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")

    # 1) Console handler at INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # 2) File handler at DEBUG level
    file_handler = logging.FileHandler(log_file, mode="w")  # Overwrite each run
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Clear any existing handlers to avoid duplicate logs (if needed)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add the two handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


"""
CLI entry point for Scene Generation Pipeline.
"""
def main():
    """
    Main function to parse arguments and dispatch subcommands.
    """

    parser = ArgumentParser(
        description="Scene Generation CLI.\n\n"
        "You can define the scene location (a rectangle) in two ways:\n"
        "  1) 'bbox' subcommand: specify four GPS corners (min_lon, min_lat, max_lon, max_lat).\n"
        "  2) 'point' subcommand: specify one GPS point, indicate its corner/center position, "
        "and give width/height in meters.\n",
        formatter_class=RawTextHelpFormatter,
    )

    # --version/-v: we'll handle printing version info ourselves after parse_args()
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version information and exit.",
    )

    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="List the available ITU materials and their frequency ranges.",
    )

    # Create a "parent" parser to hold common optional arguments.
    # Use add_help=False so we don’t duplicate the --help in child parsers.
    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output."
    )
    common_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without executing anything.",
    )
    common_parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory where scene file will be saved.",
    )
    common_parser.add_argument(
        "--osm-server-addr",
        default="https://overpass-api.de/api/interpreter",
        help="OSM server address (optional).",
    )
    common_parser.add_argument(
        "--enable-lidar-height-calibration",
        action="store_true",
        help="Enable height calibration using USGS LiDAR data.",
    )
    common_parser.add_argument(
        "--enable-building-map",
        action="store_true",
        help="Enable 2D building map output.",
    )
    common_parser.add_argument(
        "--building-data-source",
        default="overture",
        choices=["overture", "osm"],
        help=(
            "Choose which data source to use for building footprints and heights. Choices: "
            "overture, osm. Default: overture."
        ),
    )

    common_parser.add_argument(
        "--enable-lidar-terrain",
        action="store_true",
        help="Using USGS LiDAR data to generate terrain mesh.",
    )

    common_parser.add_argument(
        "--enable-dem-terrain",
        action="store_true",
        help="Using USGS 1M DEM data to generate terrain mesh.",
    )

    common_parser.add_argument(
        "--ground-material",
        default=14,
        type=int,
        help="ID of the material to use in the scene for ground. Default set to wet ground.",
    )

    common_parser.add_argument(
        "--rooftop-material",
        default=11,
        type=int,
        help="ID of the material to use in the scene for rooftops. Default set to metal.",
    )

    common_parser.add_argument(
        "--wall-material",
        default=1,
        type=int,
        help="ID of the material to use in the scene for walls. Default set to concrete.",
    )

    common_parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "If passed, sets console logging to DEBUG (file logging is always DEBUG). "
            "This overrides the default console level of INFO."
        ),
    )

    common_parser.add_argument(
        "--generate_roads",
        action="store_true",
        help="Enable road generation from Overture data. Only applicable when building_data_source is 'overture' and LiDAR and DEM terrain are disabled."
    )

    common_parser.add_argument(
        "--road-visual-material",
        type=int,
        help="ID of the material to use for visualization of roads."
    )

    # Create subparsers for different subcommands
    subparsers = parser.add_subparsers(
        title="Subcommands", dest="command", help="Available subcommands."
    )

    # Subcommand 'validate': define a bounding box by four float coordinates
    parser_validate = subparsers.add_parser(
        "validate",
        parents=[common_parser],
        help=("Validate an existing scene file"),
    )

    # Subcommand 'bbox': define a bounding box by four float coordinates
    parser_bbox = subparsers.add_parser(
        "bbox",
        parents=[common_parser],
        help=(
            "Define a bounding box using four GPS coordinates in the order: "
            "min_lon, min_lat, max_lon, max_lat."
        ),
    )
    parser_bbox.add_argument("min_lon", type=float, help="Minimum longitude.")
    parser_bbox.add_argument("min_lat", type=float, help="Minimum latitude.")
    parser_bbox.add_argument("max_lon", type=float, help="Maximum longitude.")
    parser_bbox.add_argument("max_lat", type=float, help="Maximum latitude.")

    # Subcommand 'point': define a reference point and rectangle size
    parser_point = subparsers.add_parser(
        "point",
        parents=[common_parser],
        help="Work with a single point and a rectangle size.",
    )
    parser_point.add_argument("lon", type=float, help="Longitude.")
    parser_point.add_argument("lat", type=float, help="Latitude.")
    parser_point.add_argument(
        "position",
        choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
        help="Relative position inside a rectangle.",
    )
    parser_point.add_argument("width", type=float, help="Width in meters.")
    parser_point.add_argument("height", type=float, help="Height in meters.")

    # Parse the full command line
    args = parser.parse_args()

    # Handle --version or no subcommand
    if args.version:
        print(f"{PACKAGE_NAME} version {get_package_version()}")
        sys.exit(0)

    if args.list_materials:
        print("Available ITU materials and their frequency ranges:")
        print("ID | {:^50} | Frequency Range (GHz)".format("Name", "lower", "upper"))
        for idx, item in enumerate(ITU_MATERIALS.items()):
            material, data = item
            if isinstance(data["lower_freq_limit"], list):
                for inner_idx, (low, high) in enumerate(
                    zip(data["lower_freq_limit"], data["upper_freq_limit"])
                ):
                    if inner_idx == 0:
                        print(
                            "{:<2} | {:<50} | {:^5} - {:^5}".format(
                                idx,
                                data["name"],
                                print_if_int(low / 1e9),
                                print_if_int(high / 1e9),
                            )
                        )
                    else:
                        print(
                            "{:<2} | {:<50} | {:^5} - {:^5}".format(
                                "",
                                "",
                                print_if_int(low / 1e9),
                                print_if_int(high / 1e9),
                            )
                        )

            else:
                print(
                    "{:<2} | {:<50} | {:^5} - {:^5}".format(
                        idx,
                        data["name"],
                        print_if_int(data["lower_freq_limit"] / 1e9),
                        print_if_int(data["upper_freq_limit"] / 1e9),
                    )
                )
            print("-" * 80)

        print(
            'Material properties based on\n\tITU-R P.2040-2:"Effects of building materials and structures on radiowave propagation above about 100 MHz"',
            '\n\tITU P.527-3: "Electrical characteristics of the surface of the earth"'
        )
        sys.exit(0)

    if not args.command:
        # No subcommand provided: show help and exit
        parser.print_help()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1) Set up logging by default
    #    - debug.log file captures all logs at DEBUG level
    #    - console sees INFO+ by default
    # -------------------------------------------------------------------------
    setup_logging(log_file="debug.log")

    # If user wants console debug output too, adjust console handler level.
    if args.debug:
        console_logger = logging.getLogger()  # root logger
        for handler in console_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)

    # Handle the materials
    if args.ground_material not in range(len(ITU_MATERIALS.keys())):
        logger.error(f"Invalid ground material: {args.ground_material}")
        sys.exit(1)

    if args.rooftop_material not in range(len(ITU_MATERIALS)):
        logger.error(f"Invalid rooftop material: {args.rooftop_material}")
        sys.exit(1)

    if args.wall_material not in range(len(ITU_MATERIALS)):
        logger.error(f"Invalid wall material: {args.wall_material}")
        sys.exit(1)

    if args.road_visual_material is not None and args.road_visual_material not in range(len(ITU_MATERIALS)):
        logger.error(f"Invalid road visualization material: {args.road_visual_material}")
        sys.exit(1)

    # Dispatch subcommands
    if args.command == "bbox":
        min_lon = args.min_lon
        min_lat = args.min_lat
        max_lon = args.max_lon
        max_lat = args.max_lat

        logger.info(
            f"Check the bbox at http://bboxfinder.com/#{min_lat:.{4}f},{min_lon:.{4}f},{max_lat:.{4}f},{max_lon:.{4}f}"
        )
        scene_instance = Scene()
        scene_instance(
            [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ],
            args.data_dir,
            None,
            lidar_height_calibration=args.enable_lidar_height_calibration,
            generate_building_map=args.enable_building_map,
            ground_material_type=list(ITU_MATERIALS.items())[args.ground_material][0],
            rooftop_material_type=list(ITU_MATERIALS.items())[args.rooftop_material][0],
            wall_material_type=list(ITU_MATERIALS.items())[args.wall_material][0],
            lidar_terrain=args.enable_lidar_terrain,
            dem_terrain=args.enable_dem_terrain,
            building_data_source=args.building_data_source,
            generate_roads = args.generate_roads,
            road_material_type = None if not args.road_visual_material else list(ITU_MATERIALS.items())[args.road_visual_material][0]
        )
    elif args.command == "point":
        polygon_points_gps = rect_from_point_and_size(
            args.lon, args.lat, args.position, args.width, args.height
        )
        min_lon, min_lat = polygon_points_gps[0]
        max_lon, max_lat = polygon_points_gps[2]
        logger.info(
            f"Check the bbox at http://bboxfinder.com/#{min_lat:.{4}f},{min_lon:.{4}f},{max_lat:.{4}f},{max_lon:.{4}f}"
        )
        scene_instance = Scene()
        scene_instance(
            polygon_points_gps,
            args.data_dir,
            None,
            osm_server_addr=args.osm_server_addr,
            lidar_height_calibration=args.enable_lidar_height_calibration,
            generate_building_map=args.enable_building_map,
            ground_material_type=list(ITU_MATERIALS.items())[args.ground_material][0],
            rooftop_material_type=list(ITU_MATERIALS.items())[args.rooftop_material][0],
            wall_material_type=list(ITU_MATERIALS.items())[args.wall_material][0],
            lidar_terrain=args.enable_lidar_terrain,
            dem_terrain=args.enable_dem_terrain,
            building_data_source=args.building_data_source,
            generate_roads = args.generate_roads,
            road_material_type = None if not args.road_visual_material else list(ITU_MATERIALS.items())[args.road_visual_material][0]
        )
    elif args.command == "validate":
        res_dict = {
            "scenegen_version": {
                "display_name": "Scene File Creation Tool Version",
                "value": "",
            },
            "scenegen_create_time": {
                "display_name": "Scene File Creation Time",
                "value": "",
            },
            "scenegen_min_lat": {
                "display_name": "Min Latitude",
                "value": "",
            },
            "scenegen_max_lat": {
                "display_name": "Max Latitude",
                "value": "",
            },
            "scenegen_min_lon": {
                "display_name": "Min Longitude",
                "value": "",
            },
            "scenegen_max_lon": {
                "display_name": "Max Longitude",
                "value": "",
            },
            "scenegen_center_lon": {
                "display_name": "Center Longitude",
                "value": "",
            },
            "scenegen_center_lat": {
                "display_name": "Center Latitude",
                "value": "",
            },
            "scenegen_ground_material": {
                "display_name": "Ground Materials",
                "value": "",
            },
            "scenegen_rooftop_material": {
                "display_name": "Rooftop Materials",
                "value": "",
            },
            "scenegen_wall_material": {
                "display_name": "Wall Materials",
                "value": "",
            },
            "scenegen_UTM_zone": {
                "display_name": "UTM Zone",
                "value": "",
                "comment": "",
            },
            "scenegen_bbox_width": {
                "display_name": "Bounding Box Width",
                "value": "",
                "comment": "meters (UTM Projection)",
            },
            "scenegen_bbox_length": {
                "display_name": "Bounding Box Length",
                "value": "",
                "comment": "meters (UTM Projection)",
            },
        }
        directory = args.data_dir
        if not os.path.isdir(directory):
            print(f"Error: The provided path '{directory}' is not a valid directory.")

        scene_file = os.path.join(directory, "scene.xml")
        if not os.path.isfile(scene_file):
            print(f"Error: 'scene.xml' not found in '{directory}'.")

        try:
            # Parse the XML file
            tree = ET.parse(scene_file)
            root = tree.getroot()

            # Iterate over all <default> tags
            for default_tag in root.findall(".//default"):
                key = default_tag.get("name")  # Get the attribute 'name'
                value = default_tag.get("value") if default_tag.get("value") else ""

                # If the key exists in res_dict, update the value
                if key in res_dict:
                    res_dict[key]["value"] = value

        except ET.ParseError as e:
            print(f"Error parsing XML file: {e}")

        # Define column widths
        column_width = 35  # Adjust this for better alignment

        # Print table header
        print("=" * (column_width * 2 + 5))
        print(f"{'Field'.ljust(column_width)} | {'Value'.ljust(column_width)}")
        print("=" * (column_width * 2 + 5))

        # Print each key-value pair in the dictionary
        for idx, (key, value) in enumerate(res_dict.items()):
            display_name = value["display_name"]
            display_value = value["value"] if value["value"] else "N/A"

            display_value += " " + value.get("comment", "")

            print(
                f"{display_name.ljust(column_width)} | {display_value.ljust(column_width)}"
            )
            if idx != len(res_dict) - 1:
                print("-" * (column_width * 2 + 5))

        # Print table footer
        print("=" * (column_width * 2 + 5))

        # Print the bbox link
        try:
            min_lat = float(res_dict["scenegen_min_lat"]["value"])
            max_lat = float(res_dict["scenegen_max_lat"]["value"])
            min_lon = float(res_dict["scenegen_min_lon"]["value"])
            max_lon = float(res_dict["scenegen_max_lon"]["value"])

            print(
                f"\n\nCheck the bbox at http://bboxfinder.com/#{min_lat:.{4}f},{min_lon:.{4}f},{max_lat:.{4}f},{max_lon:.{4}f}\n\n"
            )
        except Exception as e:
            # Handle the case where the bbox coordinates are not available
            pass


if __name__ == "__main__":
    main()