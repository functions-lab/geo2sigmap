# Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases

[![arXiv](https://img.shields.io/badge/arXiv-2312.14303-green?color=FF8000?color=009922)](https://arxiv.org/abs/2312.14303)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/functions-lab/geo2sigmap/blob/main/LICENSE)

Welcome to Geo2SigMap. Our repository has been divided into two primary components: 
- **Scene Generation**: A pure Python-based pipeline for generating 3D scenes for arbitrary areas of interest. This new Python-based pipeline replaces the scene generation pipeline used in our [DySPAN'24 paper](https://ieeexplore.ieee.org/document/10632773), and is more scalable, efficient, and user-friendly. See our package README (#./package/README.md)for a helpful visual of the scene generation pipeline.
- **ML-based Propagation Model**: ML-based signal coverage prediction using our pre-trained model based on the cascaded U-Net architecture, also described in our DySPAN'24 paper.

Release **v2.0.0** enhanced the original scene generation pipeline by incoporating LiDAR terrain data and downstream Digital Elevation Models (DEM). Data was sourced from the USGS 3D Elevation Program (https://www.usgs.gov/3d-elevation-program). The tool does not depend on this data for its operation. In areas with insufficient data, a flat terrain can still be assumed.

Release **v3.0.0** incorporates Overture Maps (https://overturemaps.org/) as an alternative data source option for buildings (both footprints and heights). Overture enables generation of more refined building parts and visualization of roads within scenes.

## Overview

Geo2SigMap is an efficient framework for high-fidelity RF signal mapping leveraging geographic databases, ray tracing, and a novel cascaded U-Net model. Geo2SigMap features a scalable, automated pipeline that efficiently generates 3D building and path gain (PG) maps via the integration of a suite of open-source/open-data tools including OpenStreetMap (OSM), Overture Maps, and Nvidia's Sionna Library. Geo2SigMap also features a cascaded U-Net model, which is pre-trained on pure synthetic datasets leveraging the building map and sparse signal strength (SS) map as input to predict the full SS map for the target (unseen) area. The performance of Geo2SigMap has been evaluated using large-scale field measurements collected using three types of user equipment (UE) across six LTE cells operating in the Citizens Broadband Radio Service (CBRS) band deployed on the Duke University West Campus. Our results show that Geo2SigMap achieves significantly improved root-mean-square error (RMSE) in terms of the SS map prediction accuracy compared to existing baseline methods based on channel models and ML.

## Project Structure

```sh
geo2sigmap/
├── package # Package Files
│   └── src
│   │  ├── scene_generation # Scene Generation Pipeline 
│   │   ...
│   └── README.md # Desc of Scene Gen Pipeline & Install 
├── research # ML Model and Notebook Examples
│   ├── data # Datasets
│   │   └── measurements
│   └── examples # Example Notebooks
│       └── antenna_Pattern
...

```

## Installation

Instructions for installing the Scene Generation CLI tool can be found [here](./package/README.md). We have isolated these instructions to a sub-README to avoid confusion with our notebook examples. 

Example notebooks can be found [here](./research/examples/). Each notebook example requires that you manage package dependencies **independently**. Feel free to use any environment/package manager.

## Citation

If you find Geo2SigMap useful for your research, please consider citing this paper:
```
@inproceedings{li2024geo2sigmap,
  title={Geo2SigMap: High-fidelity RF signal mapping using geographic databases},
  author={Li, Yiming and Li, Zeyu and Gao, Zhihui and Chen, Tingjun},
  booktitle={Proc. IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN)},
  year={2024}
}
```
## License

CC BY-NC 4.0

Thank you for using Geo2SigMap! If you have any questions or suggestions, feel free to open an issue on GitHub. We hope this framework accelerates your research or application in RF signal mapping.
