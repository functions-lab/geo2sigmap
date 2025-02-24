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
from .utils import rect_from_point_and_size, print_if_int
from .itu_materials import ITU_MATERIALS

try:
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except ImportError:
    # For Python < 3.8, use importlib_metadata backport
    from importlib_metadata import version as pkg_version, PackageNotFoundError
import math

PACKAGE_NAME = "scenegenerationpipe"


def get_package_version() -> str:
    """
    Attempt to retrieve the installed package version from metadata.
    Falls back to a default if the package isn't found (not installed).
    """
    try:
        return pkg_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0.dev (uninstalled)"


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


#!/usr/bin/env python3
"""
CLI entry point for Scene Generation Pipeline.
"""

import sys
from argparse import ArgumentParser, RawTextHelpFormatter


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
        "--enable-building-map",
        action="store_true",
        help="Enable 2D building map output.",
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

    # Create subparsers for different subcommands
    subparsers = parser.add_subparsers(
        title="Subcommands", dest="command", help="Available subcommands."
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
    parser_point.add_argument("lon", type=float, help="Latitude.")
    parser_point.add_argument("lat", type=float, help="Longitude.")
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
        print("ID | {:^30} | Frequency Range (GHz)".format("Name", "lower", "upper"))
        for idx, item in enumerate(ITU_MATERIALS.items()):
            material, data = item
            if isinstance(data["lower_freq_limit"], list):
                for inner_idx, (low, high) in enumerate(
                    zip(data["lower_freq_limit"], data["upper_freq_limit"])
                ):
                    if inner_idx == 0:
                        print(
                            "{:<2} | {:<20} | {:^5} - {:^5}".format(
                                idx,
                                data["name"],
                                print_if_int(low / 1e9),
                                print_if_int(high / 1e9),
                            )
                        )
                    else:
                        print(
                            "{:<2} | {:<20} | {:^5} - {:^5}".format(
                                "",
                                "",
                                print_if_int(low / 1e9),
                                print_if_int(high / 1e9),
                            )
                        )

            else:
                print(
                    "{:<2} | {:<20} | {:^5} - {:^5}".format(
                        idx,
                        data["name"],
                        print_if_int(data["lower_freq_limit"] / 1e9),
                        print_if_int(data["upper_freq_limit"] / 1e9),
                    )
                )
            print("-" * 51)

        print(
            'Material properties based on ITU-R Recommendation P.2040-2: \n\t"Effects of building materials and structures on radiowave propagation above about 100 MHz"'
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
            osm_server_addr=args.osm_server_addr,
            lidar_calibration=False,
            generate_building_map=args.enable_building_map,
            ground_material_type=list(ITU_MATERIALS.items())[args.ground_material][0],
            rooftop_material_type=list(ITU_MATERIALS.items())[args.rooftop_material][0],
            wall_material_type=list(ITU_MATERIALS.items())[args.wall_material][0],
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
            lidar_calibration=False,
            generate_building_map=args.enable_building_map,
            ground_material_type=list(ITU_MATERIALS.items())[args.ground_material][0],
            rooftop_material_type=list(ITU_MATERIALS.items())[args.rooftop_material][0],
            wall_material_type=list(ITU_MATERIALS.items())[args.wall_material][0],
        )
    else:
        # Should never happen if we covered all subcommands
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
