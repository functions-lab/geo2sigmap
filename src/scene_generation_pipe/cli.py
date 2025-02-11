"""
CLI entry point for running the core functionality of this package.
scenegenerationpipe \
    --data-dir test123 \
    --bbox -71.06025695800783 42.35128145107633 -71.04841232299806 42.35917815419112
"""

import logging

from argparse import ArgumentParser
from .core import Scene
from . import __version__



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



def main():
    parser = ArgumentParser(description="Scenen Generation Pipe.")

    
    parser.add_argument("--version", action="store_true", help="Show version and exit.")    

    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=False,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help=(
            "Four GPS coordinates defining the bounding box, in the order: "
            "min_lon, min_lat, max_lon, max_lat."
        ),
    )

    parser.add_argument(
        "--data-dir",
        required=False,
        help="Directory where data is stored or will be saved."
    )

    parser.add_argument(
        "--osm-server-addr",
        default="https://overpass-api.de/api/interpreter",
        help="OSM server address (optional)."
    )


    parser.add_argument(
        "--enable-building-map",
        action="store_true",
        help="Enable building map output (default is disabled)."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "If passed, set console logging to DEBUG (file is always at DEBUG). "
            "This overrides the default console level of INFO."
        ),
    )


    args = parser.parse_args()

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

    # -------------------------------------------------------------------------
    # 2) Handle version request
    # -------------------------------------------------------------------------
    if args.version:
        print(f"my_project version {__version__}")
        return

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
    logger.info(f"Check the bbox at http://bboxfinder.com/#{min_lat},{min_lon},{max_lat},{max_lon}")
    scene_instance = Scene()
    scene_instance([[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat], [min_lon, min_lat]], args.data_dir, None , osm_server_addr = args.osm_server_addr, lidar_calibration = False, generate_building_map=args.enable_building_map)    
    



if __name__ == "__main__":
    main()

    
