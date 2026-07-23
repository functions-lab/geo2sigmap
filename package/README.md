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
$ scenegen bbox -74.008934 40.710506 -74.002948 40.715061 --data-dir scenes/NewYork

[INFO] Check the bbox at http://bboxfinder.com/#40.7105,-74.0089,40.7151,-74.0029
[INFO] Using UTM Zone: EPSG:32618
[INFO] 
[INFO] Ground Material Type:              Wet Ground           | Frequency Range:   1   -  10   (GHz)
[INFO] Building Rooftop Material Type:    Metal                | Frequency Range:   1   -  100  (GHz)
[INFO] Building Wall Material Type:       Concrete             | Frequency Range:   1   -  100  (GHz)
[INFO] 
[INFO] Estimated ground polygon size: width=512m, height=512m
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=buildings/type=building/*
[INFO] Loaded 162 Overture building candidates
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=buildings/type=building_part/*
[INFO] Loaded 186 Overture building_part candidates
[INFO] Found 129 Overture buildings and 186 building parts before pruning
Parsing buildings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 315/315 [00:00<00:00, 391.49it/s]
```
The above commands generate a 3D scene for an area around New York City's city hall. You can preview or verify the bounding box at [bboxfinder.com](http://bboxfinder.com/#40.7105,-74.0089,40.7151,-74.0029).

Note: By default, the scene will render the terrain as a flat plane. See [Tutorial 1](#../research/examples/1_sionna_coverage_map.ipynb) to include elevation and terrain information.

Choose the building data source mode with `--building-data-source`:

- `overture`: Footprints and heights are sourced from Overture buildings and building parts. Heights of buildings and parts are determined according to the following hierarchy: Overture `height` tag, Overture `num_floors` tag * floor height multiplier, LiDAR HAG sampling and averaging if the `enable-lidar-height-calibration` is also used, and then random Gaussian fallback. If `min_height` or, secondarily, `min_floor` is available in Overture, building/part height is offset accordingly. Pruning of buildings with associated parts and detection/correction of some common Overture contributor misinterpretations is performed. `overture` is the default selection for building data source.
- `osm`: Footprints and heights are sourced from OSM buildings. Heights of buildings are determined accoridng to the following hierarchy: LiDAR HAG sampling and averaging if the `enable-lidar-height-calibration` is also used, OSM `building:height` tag, OSM `height` tag, OSM `building:levels` tag, OSM `levels` tag, and then random Gaussian fallback.

### 2) Generate 3D Scene using One Point + Rectangle Dimension
```console
$ scenegen point -74.0059413 40.7127837 center 500 500  --data-dir scenes/NewYork_center

[INFO] Check the bbox at http://bboxfinder.com/#40.7106,-74.0089,40.7150,-74.0029
[INFO] Using UTM Zone: EPSG:32618
[INFO] 
[INFO] Ground Material Type:              Wet Ground           | Frequency Range:   1   -  10   (GHz)
[INFO] Building Rooftop Material Type:    Metal                | Frequency Range:   1   -  100  (GHz)
[INFO] Building Wall Material Type:       Concrete             | Frequency Range:   1   -  100  (GHz)
[INFO] 
[INFO] Estimated ground polygon size: width=501m, height=501m
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=buildings/type=building/*
[INFO] Loaded 162 Overture building candidates
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=buildings/type=building_part/*
[INFO] Loaded 186 Overture building_part candidates
[INFO] Found 124 Overture buildings and 179 building parts before pruning
Parsing buildings: 100%|████████████████████████████████████████████████████| 303/303 [00:00<00:00, 419.97it/s]
```

Note: The public overpass-api.de server imposes query rate limits (~2–10 queries/sec). For higher throughput (e.g., ~100–200 queries/sec on an SSD machine), consider [hosting your own OSM server](https://wiki.openstreetmap.org/wiki/Overpass_API/Installation).

### 3) Generate 3D Scene using LiDAR or DEM Data for Terrain or Height Calibration

This can be done by following the structure of the previous command(s) while adding additional flags. You can generate the scene using LiDAR data (`--enable-lidar-terrain`) or DEM data (both `--enable-lidar-terrain` and `--enable-dem-terrain`):

```console
$ scenegen bbox -74.008934 40.710506 -74.002948 40.715061 --data-dir scenes/NewYork_lidar --enable-lidar-terrain

[INFO] Check the bbox at http://bboxfinder.com/#40.7105,-74.0089,40.7151,-74.0029
[INFO] Using UTM Zone: EPSG:32618
[INFO] 
[INFO] Ground Material Type:              Wet Ground           | Frequency Range:   1   -  10   (GHz)
[INFO] Building Rooftop Material Type:    Metal                | Frequency Range:   1   -  100  (GHz)
[INFO] Building Wall Material Type:       Concrete             | Frequency Range:   1   -  100  (GHz)
[INFO] 
Loading local 3DEP dataset polygons...
Done. 3DEP polygons downloaded and projected to  EPSG:32618
Area of Interest: POLYGON ((-8238803.4366509635 4969567.452677717, -8238803.4366509635 4970570.884914368, -8237803.898943133 4970570.884914368, -8237803.898943133 4969567.452677717, -8238803.4366509635 4969567.452677717))
NY_NewYorkCity
https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_NewYorkCity/ept.json
Found 1 intersecting datasets
Successfully generated HAG data
Checking lidar_terrain.ply
generate_terrain_mesh
True
<ScaledArrayView([584049.66 583863.32 583867.42 ... 583591.05 583585.68 583582.4 ])>
<ScaledArrayView([4507458.14 4507270.14 4507085.54 ... 4507709.41 4507712.17 4507714.92])>
[584049.66 583863.32 583867.42 ... 583591.05 583585.68 583582.4 ]
[4507458.14 4507270.14 4507085.54 ... 4507709.41 4507712.17 4507714.92]
centerx, y 583964.3852012418 4507349.227429416
/home/rt279/miniconda3/envs/g2sm/lib/python3.12/site-packages/pyvista/core/pointset.py:1386: PyVistaDeprecationWarning: The current behavior of `pv.PolyData.n_faces` has been deprecated.
                Use `pv.PolyData.n_cells` or `pv.PolyData.n_faces_strict` instead.
                See the documentation in '`pv.PolyData.n_faces` for more information.
  warnings.warn(
Ori # of faces:  33475
pro_decimated # of faces:  3347
[INFO] Estimated ground polygon size: width=512m, height=512m
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=buildings/type=building/*
[INFO] Loaded 162 Overture building candidates
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=buildings/type=building_part/*
[INFO] Loaded 186 Overture building_part candidates
[INFO] Found 129 Overture buildings and 186 building parts before pruning
Parsing buildings: 100%|████████████████████████████████████████████████████| 315/315 [00:01<00:00, 206.18it/s]
```

Separately, you can toggle LiDAR inclusion in the building height determination hierarchy with the `--enable-lidar-height-calibration` flag. When the building data source is `osm`, LiDAR random sampling and averaging over each footprint is used as the first step in the hierarchy. When the building data source is `overture`, LiDAR random sampling and averaging over each footprint is used after referencing Overture's `height` and `num_floors` fields, but before random fallback.

```
console
$ scenegen bbox -74.008934 40.710506 -74.002948 40.715061 --data-dir scenes/NewYork_osm_lidar --enable-lidar-height-calibration --building-data-source osm

[INFO] Check the bbox at http://bboxfinder.com/#40.7105,-74.0089,40.7151,-74.0029
[INFO] Using UTM Zone: EPSG:32618
[INFO] 
[INFO] Ground Material Type:              Wet Ground           | Frequency Range:   1   -  10   (GHz)
[INFO] Building Rooftop Material Type:    Metal                | Frequency Range:   1   -  100  (GHz)
[INFO] Building Wall Material Type:       Concrete             | Frequency Range:   1   -  100  (GHz)
[INFO] 
Loading local 3DEP dataset polygons...
Done. 3DEP polygons downloaded and projected to  EPSG:32618
Area of Interest: POLYGON ((-8238803.4366509635 4969567.452677717, -8238803.4366509635 4970570.884914368, -8237803.898943133 4970570.884914368, -8237803.898943133 4969567.452677717, -8238803.4366509635 4969567.452677717))
NY_NewYorkCity
https://s3-us-west-2.amazonaws.com/usgs-lidar-public/NY_NewYorkCity/ept.json
Found 1 intersecting datasets
Successfully generated HAG data
[INFO] Estimated ground polygon size: width=512m, height=512m
Parsing buildings: 100%|████████████████████████████████████████████████████| 165/165 [00:00<00:00, 470.58it/s]
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

### 5) Generate Roads When Building Data Source is Overture
You can generate roads when using Overture as the building data source and LiDAR/DEM terrain are disabled. Use the `--generate-roads` flag, and specify a material type with `--road-visual-material` (followed by a `<MATERIAL_ID>`) that is visually distinct from the ground material (by default, the road visual material will be the same as the ground material and therefore not visible).

```
console
$ scenegen bbox -74.008934 40.710506 -74.002948 40.715061 --data-dir scenes/NewYork_with_roads --building-data-source overture --generate-roads --road-visual-material 7

[INFO] Check the bbox at http://bboxfinder.com/#40.7105,-74.0089,40.7151,-74.0029
[INFO] Using UTM Zone: EPSG:32618
[INFO] 
[INFO] Ground Material Type:              Wet Ground           | Frequency Range:   1   -  10   (GHz)
[INFO] Building Rooftop Material Type:    Metal                | Frequency Range:   1   -  100  (GHz)
[INFO] Building Wall Material Type:       Concrete             | Frequency Range:   1   -  100  (GHz)
[INFO] 
[INFO] Estimated ground polygon size: width=512m, height=512m
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=buildings/type=building/*
[INFO] Loaded 162 Overture building candidates
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=buildings/type=building_part/*
[INFO] Loaded 186 Overture building_part candidates
[INFO] Found 129 Overture buildings and 186 building parts before pruning
Parsing buildings: 100%|████████████████████████████████████████████████████| 315/315 [00:00<00:00, 410.20it/s]
Using Overture release: s3://overturemaps-us-west-2/release/2026-07-22.0/theme=transportation/type=segment/*
[INFO] Loaded 444 Overture road segment candidates
[INFO] Using 444 Overture road segments
Parsing roads: 100%|███████████████████████████████████████████████████████| 444/444 [00:00<00:00, 1476.05it/s]
```

### 6) Preview 3D Scene in Sionna

After the above example command, the 3D scene file is saved to the corresponding folder under `./scenes/`. You can load it directly in Sionna to explore or run ray tracing simulations. Please refer to [Tutorial #1](../research/examples/1_sionna_coverage_map.ipynb) and [Tutorial #2](../research/examples/2_sionna_rays_analysis.ipynb) for two example notebooks.

## Visualizing the Scene Generation Pipeline
```
                              GPS bounding box
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
 building_data_source?        USGS LiDAR query          Terrain query
          │                          │               (LiDAR or DEM)
     ┌────┴────┐                     │                          │
     │         │                     │                          ▼
     ▼         ▼                     ├──────────────► terrain mesh (PLY)
 OSM query  Overture query           │
     │         │                     └──────────────► HAG point cloud
     │         │
     │         ▼
     │   Parent/part resolution
     │   & pruning
     │         │
     └────┬────┘
          ▼
   Building footprints
          │
          ▼
   Height determination
          │
    ┌─────┴───────────────────────────────────────────────┐
    │                                                     │
    │ OSM hierarchy                                       │
    │   1. LiDAR HAG sampling (optional)                  │
    │   2. building:height                                │
    │   3. height                                         │
    │   4. building:levels × 3.5 m                        │
    │   5. levels × 3.5 m                                 │
    │   6. random fallback                                │
    │                                                     │
    │ Overture hierarchy                                  │
    │   1. height                                         │
    │   2. num_floors × 3.5 m                             │
    │   3. LiDAR HAG sampling (optional)                  │
    │   4. random fallback                                │
    └─────────────────────────────────────────────────────┘
          │
          ▼
  (Overture only)
  • minimum-height offset
  • contributor discrepancy calibration
          │
          ▼
 Extruded building meshes (PLY)
          │
          ├──────────────────────────────────────┐
          │                                      │
          ▼                                      ▼
 Combine with ground mesh                Building Height Map (.npy)
 (terrain mesh if enabled)
          │
          ▼
      scene.xml
(geometry + materials + metadata)
```
