# Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases

Welcome to the Geo2SigMap, this is the first work that: 
* Designs an automated framework that integrates open-source tools, including geographic databases (OSM), computer graphics (Blender), and ray tracing (Sionna), and supports scalable ray tracing and RF signal mapping at-scale using real-world building information;
* Develops a novel cascaded U-Net architecture that achieves significantly improved signal strength (SS) map prediction accuracy compared to existing baseline methods based on channel models and ML.

## Note: Current branch is under code refactoring. Only the 3D scene generation pipeline is available. ML part will be published later.
## TABLE OF CONTENTS
1. [Overview](#overview)
<!-- 2. [Repo Structure](#repo-structure) -->
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

<!-- This is an active project, if you are interested to have a community discussion, please start a new discussion thread in the discussion tab and we will get back to you as soon as possible. -->

<!-- ## Repo Structure

|  Source Files      |  Description                                                                                                             |
|  -----             |  -----                                                                                                                   |
|  `gen_data/`   |  This folder contains the code of our pipeline to generate signal coverage map based on realworld building infos. |
|  `data/`    |  This folder contains the sample data generate by our pipeline.                                                 |
|  `ml/`       |  This folder contains the code of cascade machine learning models.                                                           | -->

## Installation

```bash
git clone -b new_pipe https://github.com/functions-lab/geo2sigmap
cd geo2sigmap
pip install .
```

<!-- ### Docker

Run the following command to use our pre-compiled docker image:
```console
docker run --name g2s -it ffkshakf/geo2sigmap:latest bash
```

We only provide an amd64 arch docker image. If you run on a Apple Silicon or ARM64, you can add `--platform linux/amd64` to run it under emulation.
```console
docker run --name g2s --platform linux/amd64 -it ffkshakf/geo2sigmap:latest bash
``` -->
<!-- 
### Install from Scratch
Note: The following commands have been tested on a clean install Ubuntu:22.04, you can follow the Blender offcial document about compiling Blender [here](https://wiki.blender.org/wiki/Building_Blender) if you would like to run on other OS/ARCH. -->

<!-- #### Install Initial Packages
```console
sudo apt update
sudo apt install python3 python3-pip git
pip install --target=/usr/lib/python3/dist-packages Cython
``` -->
<!-- 
#### Clone the Blender Source Code and Apply Changes
```console
mkdir ~/blender-git
cd ~/blender-git
git clone --depth 1 --branch v3.3.1 https://projects.blender.org/blender/blender.git
cd blender
git switch -c v3.3.1_geo2sigmap
perl -i -pe 'BEGIN{undef $/;} s/  bool is_output_operation\(bool \/\*rendering\*\/\) const override\n  \{\n    if \(G.background\) \{\n      return false;\n    \}\n    return is_active_viewer_output\(\);\n  \}/  bool is_output_operation\(bool \/\*rendering\*\/\) const override\n  \{\n    return is_active_viewer_output\(\);\n  \}/sm' source/blender/compositor/operations/COM_ViewerOperation.h
perl -i -pe 'BEGIN{undef $/;} s/  bool is_output_operation\(bool \/\*rendering\*\/\) const override\n  \{\n    return !G.background;\n  \}/  bool is_output_operation\(bool \/\*rendering\*\/\) const override\n  \{\n    return true;\n  \}/sm' source/blender/compositor/operations/COM_PreviewOperation.h
perl -i -pe 's/(_src=\$SRC\/)USD-\$USD_VERSION/$1OpenUSD-\$USD_VERSION/' ./build_files/build_environment/install_deps.sh
perl -i -pe 'BEGIN{undef $/;} s/(patch -d \$_src -p1 < \$SCRIPT_DIR\/patches\/usd\.diff\n\s+fi\n\n\s+cd \$_src)/$1\n\n    sed -i.bak  -e '\''s\/\.if !defined\.ARCH_OS_WINDOWS\.\/#if 0\/'\'' -e '\''s\/\.if defined\.ARCH_COMPILER_GCC\..*\/#if 0\/'\'' -e '\''s\/defined\.ARCH_COMPILER_CLANG\.\/\/'\'' -e '\''s\/\.if defined\.ARCH_OS_LINUX\.\/#if 0\/'\'' -e '\''s\/\.if !defined\.ARCH_OS_LINUX\.\/#if 1\/'\'' pxr\/base\/arch\/mallocHook.cpp/' ./build_files/build_environment/install_deps.sh
``` -->
<!-- 
#### Install Blender Libraries & Compile Blender
```console 
bash ./build_files/build_environment/install_deps.sh --with-embree --with-oidn
```

When the script complete you will see some thing like the follow:
```
Or even simpler, just run (in your blender-source dir):
  make -j20 BUILD_CMAKE_ARGS="-U XXXXXXXXXXXXXXXX

Or in all your build directories:
  cmake -U *SNDFILE* -U PYTHON* -U XXXXXXXXXXXXXXXX

```
Copy the first section and add an "release" between "make" and "-j" like the following ones and then execute it.

```console 
make update
make release -j20 BUILD_CMAKE_ARGS="-U XXXXXXXXXXXXXXXX
``` -->

<!-- #### Download Blender Add-on & Apply Changes
There are two add-ons, [Blosm](https://prochitecture.gumroad.com/l/blender-osm) and [mitsuba-blender](https://github.com/mitsuba-renderer/mitsuba-blender/releases/). Download the zip file and place them in the root of this project `*/geo2sigmap/`.
```console 
unzip mitsuba-blender.zip
perl -i -pe 's/  result = subprocess\.run\(\[sys\.executable, '-m', 'ensurepip'\], capture_output=True\)\n\s+return result\.returncode == 0/return True/' mitsuba-blender/__init__.py
zip -r -0 mitsuba-blender.zip mitsuba-blender
```

#### Install Sionna
Please follow [Sionna's official document](https://nvlabs.github.io/sionna/installation.html) to install Sionna.

#### Install Pytorch

Please follow [Pytorch's official document](https://pytorch.org/get-started/locally/) to install PyTorch. -->

## Example Usage

### Generate 3D Scene
Simply run:
```bash
$ scenegenerationpipe \
    --data-dir DavisonHall \
    --bbox -71.06025695800783 42.35128145107633 -71.04841232299806 42.35917815419112

2025-02-10 13:29:46,052 - scene_generation_pipe.cli - [INFO] - Check the bbox at http://bboxfinder.com/#42.35128145107633,-71.06025695800783,42.35917815419112,-71.04841232299806
2025-02-10 13:29:46,110 - scene_generation_pipe.core - [INFO] - For the given bbox, using UTM area: EPSG:32619
center_x_polygon,center_y_polygon 330811.0120150992 4691262.171513598
center_x, center_y 330810.9819912539 4691262.180812276
ground_polygon_bbox (330312.62209543784, 4690811.958469893, 331309.3418870699, 4691712.403154659)
2025-02-10 13:29:46,111 - scene_generation_pipe.core - [INFO] - Estimated ground coverage: width=997m, height=901m
tmpres (330312.62209543784, 4691712.403154659)
Parsing buildings: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 389/389 [00:00<00:00, 1403.12it/s]
```
The above command will generate the 3D Scene for Davasion Hall in Duke Camups. You can check the area location by the above link.

You can check the detail option by the `-h` arguments.


Note: The public OSM sever have a query limitation around 2-10 query/second, so if you want to achieve a faster process speed, consider deploy a self host OSM server following the OSM offcial document [here](https://wiki.openstreetmap.org/wiki/Overpass_API/Installation). A reasonable speed of self hosted server would be around 100-200 query/second on a SSD computer.

<!--- #### Generate signal coverage map using Sionna
To use sionna generate signal coverage map, run xxxx. The sionna cofigue is defined in xxxx.

#### Train the model
To train our model, run xxxxx. ---> 

## License

Distributed under the APACHE LICENSE, VERSION 2.0
