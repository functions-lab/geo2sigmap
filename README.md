# Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases

Welcome to the Geo2SigMap, this is the first work that: 
* Designs an automated framework that integrates open-source tools, including geographic databases (OSM), computer graphics (Blender), and ray tracing (Sionna), and supports scalable ray tracing and RF signal mapping at-scale using real-world building information;
* Develops a novel cascaded U-Net architecture that achieves significantly improved signal strength (SS) map prediction accuracy compared to existing baseline methods based on channel models and ML.

## Note: Current branch is under code refactoring. Only the 3D scene generation pipeline is available. ML part will be published later.
## TABLE OF CONTENTS
1. [Overview](#overview)
2. [Installation](#installation)
3. [Example Usage](#example-usage)
4. [License](#license)

## Overview

Geo2SigMap is an efficient framework for high-fidelity RF signal mapping leveraging geographic databases, ray tracing, and a novel cascaded U-Net model. Geo2SigMap features an automated pipeline that efficiently generates 3D building and path gain (PG) maps via the integration of a suite of open-sourced tools, including OpenStreetMap (OSM), Blender, and Sionna. Geo2SigMap also features a cascaded U-Net model, which is pre-trained on pure synthetic datasets leveraging the building map and sparse SS map as input to predict the full SS map for the target (unseen) area. The performance of Geo2SigMap has been evalauted using large-scale field measurements collected using three types of user equipment (UE) across six LTE cells operating in the citizens broadband radio service (CBRS) band deployed on the Duke University West Campus. Our results show that Geo2SigMap achieves significantly improved root-mean-square error (RMSE) in terms of of the SS map prediction accuracy compared to existing baseline methods based on channel models and ML.

If you find Geo2SigMap useful for your research, please consider citing:
```
@article{li2023geo2sigmap,
  title = {{Geo2SigMap}: High-Fidelity {RF} Signal Mapping Using Geographic Databases},
  author = {Li, Yiming and Li, Zeyu and Gao, Zhihui and Chen, Tingjun},
  journal={arXiv:2312.14303},
  year={2023}
}

```

## Installation

#### Dependency
* Python >= 3.9
  
```bash
git clone -b new_pipe https://github.com/functions-lab/geo2sigmap
cd geo2sigmap
python3 -m pip install .
```


## Example Usage


### Generate 3D Scene
Below are examples showing how to generate a 3D scene for your chosen location. There are two ways to define the bounding box (scene area):

1. Directly specify four GPS corners.
2. Provide one GPS point, indicate its position in the rectangle (top-left, bottom-right, etc.), and supply width and height in meters.

To see all available options for scene generation, use `-h`:
```console
$ scenegen -h
usage: scenegenerationpipe [-h] [--version] {bbox,point} ...

Scene Generation CLI.

You can define the scene location (a rectangle) in two ways:
  1) 'bbox' subcommand: specify four GPS corners (min_lon, min_lat, max_lon, max_lat).
  2) 'point' subcommand: specify one GPS point, indicate its corner/center position, and give width/height in meters.

options:
  -h, --help     show this help message and exit
  --version, -v  Show version information and exit.

Subcommands:
  {bbox,point}   Available subcommands.
    bbox         Define a bounding box using four GPS coordinates in the order: min_lon, min_lat, max_lon, max_lat.
    point        Work with a single point and a rectangle size.
```

#### 1) Generate 3D Scene via Four Corner Points
```console
$ scenegen bbox -71.0602 42.3512 -71.0484 42.3591 --data-dir Boston

[INFO] Check the bbox at http://bboxfinder.com/#42.3512,-71.0602,42.3591,-71.0484
[INFO] Using UTM Zone: EPSG:32619
[INFO] Estimated ground coverage: width=994m, height=901m
Parsing buildings: 100%|█████████████████████████████████████████████████████████████| 389/389 [00:00<00:00, 1403.12it/s]
```

#### 2) Generate 3D Scene via One Point + Rectangle Dimensions
```console
$ scenegen point -71.0550 42.3566 top-left 997 901 --data-dir Boston_top-left

[INFO] Check the bbox at http://bboxfinder.com/#42.34849072135817,-71.05473573275874,42.356816293828466,-71.04290140697601
[INFO] Using UTM Zone: EPSG:32619
[INFO] Estimated ground coverage: width=997m, height=902m
Parsing buildings: 100%|█████████████████████████████████████████████████████████████| 168/168 [00:00<00:00, 1383.61it/s]
```
The above commands generate a 3D scene for an area in downtown Boston. You can preview or verify the bounding box at [bboxfinder.com](http://bboxfinder.com/#42.35128145107633,-71.06025695800783,42.35917815419112,-71.04841232299806).


### Preview 3D Scene in Sionna


After above example command, the 3D scene file is located in the `Boston` folder. You can load it directly in Sionna to explore or run ray tracing simulations. For a working example, see the [examples/sionna_rt_coverage_map.ipynb](examples/SionnaRayTracing.ipynb) notebook.


For additional details, refer to the [Sionna RT documentation](https://nvlabs.github.io/sionna/api/rt.html).




Note: The public overpass-api.de server imposes query rate limits (~2–10 queries/sec). For higher throughput (e.g., ~100–200 queries/sec on an SSD machine), consider [hosting your own OSM server](https://wiki.openstreetmap.org/wiki/Overpass_API/Installation).



## License

Distributed under the APACHE LICENSE, VERSION 2.0

Thank you for using Geo2SigMap! If you have any questions or suggestions, feel free to open an issue on GitHub. We hope this framework accelerates your research or application in RF signal mapping.