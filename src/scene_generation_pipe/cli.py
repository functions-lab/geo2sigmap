"""
CLI entry point for running the scene generation functionality.

Users can define the scene location by a rectangle in two ways:
1) Specifying four GPS corners directly, or
2) Providing one reference point plus width/height in meters and a corner/center position.

scenegenerationpipe \
    --data-dir test123 \
    --bbox -71.06025695800783 42.35128145107633 -71.04841232299806 42.35917815419112
"""

import logging

from argparse import ArgumentParser
from .core import Scene

try:
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except ImportError:
    # For Python < 3.8, use importlib_metadata backport
    from importlib_metadata import version as pkg_version, PackageNotFoundError

PACKAGE_NAME = "scenegenerationpipe"  # <-- Replace with your actual package name in pyproject.toml

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

    # 1) Console handler at INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

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


# def main():
#     parser = ArgumentParser(description="Scenen Generation Pipe.")


#     parser.add_argument("- v, --version", action="store_true", help="Show version and exit.")

#         # Create a "parent" parser for common optional arguments.
#     # Use `add_help=False` so we don’t get a second --help in the child parser.
#     common_parser = ArgumentParser(add_help=False)
#     common_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
#     common_parser.add_argument("--dry-run", action="store_true", help="Simulate actions without executing")


#     common_parser.add_argument(
#         "--data-dir",
#         required=False,
#         help="Directory where data is stored or will be saved."
#     )

#     common_parser.add_argument(
#         "--osm-server-addr",
#         default="https://overpass-api.de/api/interpreter",
#         help="OSM server address (optional)."
#     )

#     common_parser.add_argument(
#         "--enable-building-map",
#         action="store_true",
#         help="Enable building map output (default is disabled)."
#     )
#     common_parser.add_argument(
#         "--debug",
#         action="store_true",
#         help=(
#             "If passed, set console logging to DEBUG (file is always at DEBUG). "
#             "This overrides the default console level of INFO."
#         ),
#     )


#     # Create subparsers
#     subparsers = parser.add_subparsers(
#         title="Subcommands",
#         dest="command",
#         help="Available subcommands"
#     )

#     # Subcommand 'A'
#     parser_a = subparsers.add_parser(
#         "bbox",
#         parents=[common_parser],
#         help="Four GPS coordinates defining the bounding box, in the order: "
#              "min_lon, min_lat, max_lon, max_lat."
#     )
#     parser_a.add_argument("min_lon", type=float, help="min_lon")
#     parser_a.add_argument("min_lat", type=float, help="min_lat")
#     parser_a.add_argument("max_lon", type=float, help="max_lon")
#     parser_a.add_argument("max_lat", type=float, help="max_lat")

#     # Subcommand 'B'
#     parser_b = subparsers.add_parser(
#         "point",
#         parents=[common_parser],
#         help="Perform action B"
#     )
#     parser_b.add_argument("lat",  help="latitude")
#     parser_b.add_argument("lon", help="longtitude")
#     parser_b.add_argument(
#         "position",
#         choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
#         help="Position relative to a rectangle"
#     )
#     parser_b.add_argument("width",  help="Width in meters")
#     parser_b.add_argument("height", help="Height in meters")


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
        required=False,
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
    parser_point.add_argument("lat", type=float, help="Latitude.")
    parser_point.add_argument("lon", type=float, help="Longitude.")
    parser_point.add_argument(
        "position",
        choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
        help="Relative position inside a rectangle.",
    )
    parser_point.add_argument("width", type=float, help="Width in meters.")
    parser_point.add_argument("height", type=float, help="Height in meters.")

    # def handle_bbox(args):
    #     """
    #     Handle the 'bbox' subcommand logic here.
    #     """
    #     # Example: just show the parsed values
    #     if args.verbose:
    #         print("[DEBUG] BBOX subcommand with verbose output.")
    #     print(f"BBOX coordinates: min_lon={args.min_lon}, min_lat={args.min_lat}, "
    #           f"max_lon={args.max_lon}, max_lat={args.max_lat}")
    #     # ... do the actual bounding box tasks ...

    # def handle_point(args):
    #     """
    #     Handle the 'point' subcommand logic here.
    #     """
    #     if args.verbose:
    #         print("[DEBUG] POINT subcommand with verbose output.")
    #     print(f"POINT command: lat={args.lat}, lon={args.lon}, position={args.position}, "
    #           f"width={args.width}m, height={args.height}m")
    #     # ... do the actual point tasks ...

    # if __name__ == "__main__":
    #     main()

    # parser.add_argument(
    #     "--bbox",
    #     nargs=4,
    #     type=float,
    #     required=False,
    #     metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    #     help=(
    #         "Four GPS coordinates defining the bounding box, in the order: "
    #         "min_lon, min_lat, max_lon, max_lat."
    #     ),
    # )

    # Parse the full command line
    args = parser.parse_args()
    
    # Handle --version or no subcommand
    if args.version:
        print(f"{PACKAGE_NAME} version {get_package_version()}")
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



    #     # Dispatch subcommands
    # if args.command == "bbox":
    #     handle_bbox(args)
    # elif args.command == "point":
    #     handle_point(args)
    # else:
    #     # Should never happen if we covered all subcommands
    #     parser.print_help()
    #     sys.exit(1)

    # -------------------------------------------------------------------------
    # 3) Validate mandatory arguments if not in --version mode
    # -------------------------------------------------------------------------
    if not args.bbox or not args.data_dir:
        parser.error(
            "You must specify --bbox and --data-dir (unless you are using --version)."
        )

    # -------------------------------------------------------------------------
    # 4) Extract parameters
    # -------------------------------------------------------------------------
    min_lon, min_lat, max_lon, max_lat = args.bbox
    logger = logging.getLogger(__name__)
    logger.info(
        f"Check the bbox at http://bboxfinder.com/#{min_lat},{min_lon},{max_lat},{max_lon}"
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
    )


if __name__ == "__main__":
    main()
