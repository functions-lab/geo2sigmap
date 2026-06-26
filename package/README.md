## Installation

**1. Create the Conda Environment**
```bash
conda create --yes --name g2sm --channel conda-forge pdal python=3.12
conda activate g2sm
pip install pyvista==0.45.2
```

**2. Clone and Install geo2sigmap**:
```bash
git clone https://github.com/functions-lab/geo2sigmap
cd geo2sigmap/package
pip install .
```
The tutorial below demonstrates the capabilities of the scene generation pipeline.

## Tutorial: CLI Tool

There are two ways to define a bounding box (scene area):

1. Directly specify four GPS corners.
2. Provide one GPS point, indicate its position in the rectangle (top-left, bottom-right, etc.), and supply width and height in meters.

To see all available options for scene generation, use `-h`:

### 1) Generate 3D Scene using Four Corner Points

```console
$ scenegen bbox -71.0602 42.3512 -71.0484 42.3591 --data-dir scenes/Boston

[INFO] Check the bbox at http://bboxfinder.com/#42.3512,-71.0602,42.3591,-71.0484
[INFO] Using UTM Zone: EPSG:32619
[INFO] 
[INFO] Ground Material Type:           Wet Ground       | Frequency Range:   1   -  10   (GHz)
[INFO] Building Rooftop Material Type: Metal            | Frequency Range:   1   -  100  (GHz)
[INFO] Building Wall Material Type:    Concrete         | Frequency Range:   1   -  100  (GHz)
[INFO] 
[INFO] Estimated ground polygon size: width=994m, height=901m
Parsing buildings: 100%|█████████████████████| 389/389 [00:00<00:00, 1403.12it/s]
```
The above commands generate a 3D scene for an area in downtown Boston. You can preview or verify the bounding box at [bboxfinder.com](http://bboxfinder.com/#42.3512,-71.0602,42.3591,-71.0484).

Note: By default, the scene will render the terrain as a flat plane. See [Tutorial 3](#3-generate-3d-scene-using-lidar-or-dem-data) to include elevation and terrain information.

Choose the building height source mode with `--building-height-mode`:

- `lidar-osm` or `1`: LiDAR HAG sampling and averaging, OSM explicit height tags (`building:height`, then `height`), OSM floor-count tags (`building:levels`, then `levels`), then random Gaussian fallback. This is the default.
- `overture` or `2`: Overture footprints with Overture BuildingParts preferred when available for more detailed geometry. Heights use Overture exact height, Overture levels times floor height, LiDAR HAG sampling and averaging, then random Gaussian fallback; part base offsets use Overture `min_height` or `min_floor` when present. This mode does not query OSM buildings.

### 2) Generate 3D Scene using One Point + Rectangle Dimension
```console
$ scenegen point -71.0550 42.3566 top-left 997 901 --data-dir scenes/Boston_top-left

[INFO] Check the bbox at http://bboxfinder.com/#42.3485,-71.0547,42.3568,-71.0429
[INFO] Using UTM Zone: EPSG:32619
[INFO] 
[INFO] Ground Material Type:           Wet Ground       | Frequency Range:   1   -  10   (GHz)
[INFO] Building Rooftop Material Type: Metal            | Frequency Range:   1   -  100  (GHz)
[INFO] Building Wall Material Type:    Concrete         | Frequency Range:   1   -  100  (GHz)
[INFO] 
[INFO] Estimated ground polygon size: width=997m, height=902m
Parsing buildings: 100%|█████████████████████| 168/168 [00:00<00:00, 1383.61it/s]
```

Note: The public overpass-api.de server imposes query rate limits (~2–10 queries/sec). For higher throughput (e.g., ~100–200 queries/sec on an SSD machine), consider [hosting your own OSM server](https://wiki.openstreetmap.org/wiki/Overpass_API/Installation).

### 3) Generate 3D Scene using LiDAR or DEM Data

This can be done by following the structure of the previous command(s) while adding additional flags. You can generating the scene using LiDAR data, DEM data or both:

```console
$ scenegen bbox -71.0602 42.3512 -71.0484 42.3591 --data-dir scenes/Boston_dem --enable-lidar-terrain --enable-dem-terrain

[INFO] Check the bbox at http://bboxfinder.com/#42.3485,-71.0547,42.3568,-71.0429
[INFO] Using UTM Zone: EPSG:32619
[INFO] 
[INFO] Ground Material Type:           Wet Ground       | Frequency Range:   1   -  10   (GHz)
[INFO] Building Rooftop Material Type: Metal            | Frequency Range:   1   -  100  (GHz)
[INFO] Building Wall Material Type:    Concrete         | Frequency Range:   1   -  100  (GHz)
[INFO] 
[INFO] Estimated ground polygon size: width=997m, height=902m
Parsing buildings: 100%|█████████████████████| 168/168 [00:00<00:00, 1383.61it/s]
```


### 4) Customize Material Types for Ground, Building Rooftops/Walls
You can specify material types for different surfaces using the following arguments: `--ground-material`, `--rooftop-material`, and `--wall-material` followed by a `<MATERIAL_ID>`. List all available materials and their properties using:
```console
$ scenegen --list-materials

Available ITU materials and their frequency ranges:
ID |         Name         | Frequency Range (GHz)
0  | Vacuum (≈Air)        | 0.001 -  100 
---------------------------------------------------
1  | Concrete             |   1   -  100 
---------------------------------------------------
2  | Brick                |   1   -  40  
---------------------------------------------------
3  | Plasterboard         |   1   -  100 
---------------------------------------------------
4  | Wood                 | 0.001 -  100 
---------------------------------------------------
5  | Glass                |  0.1  -  100 
   |                      |  220  -  450 
---------------------------------------------------
6  | Ceiling Board        |   1   -  100 
   |                      |  220  -  450 
---------------------------------------------------
7  | Chipboard            |   1   -  100 
---------------------------------------------------
8  | Plywood              |   1   -  40  
---------------------------------------------------
9  | Marble               |   1   -  60  
---------------------------------------------------
10 | Floorboard           |  50   -  100 
---------------------------------------------------
11 | Metal                |   1   -  100 
---------------------------------------------------
12 | Very Dry Ground      |   1   -  10  
---------------------------------------------------
13 | Medium Dry Ground    |   1   -  10  
---------------------------------------------------
14 | Wet Ground           |   1   -  10  
---------------------------------------------------
Material properties based on ITU-R Recommendation P.2040-2: 
        "Effects of building materials and structures on radiowave propagation above about 100 MHz"
```

### 5) Preview 3D Scene in Sionna

After the above example command, the 3D scene file is saved to the corresponding folder under `./scenes/`. You can load it directly in Sionna to explore or run ray tracing simulations. Please refer to [Tutorial #1](examples/sionna_rt_coverage_map.ipynb) and [Tutorial #2](examples/sionna_rt_rays_analyze.ipynb) for two example notebooks.
