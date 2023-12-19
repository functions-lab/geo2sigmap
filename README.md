# GEO2SIGMAP: High-Fidelity RF Signal Mapping Using Geographic Databases

Welcome to the Geo2SigMap, the first work that: 
* (i) Develops an automated framework integrating open-source geographic databases, computer graphics, and ray tracing tools to perform a massive raytracing based on the realworld building information.
* (ii) Integrates a novel cascaded U-Net architecture that achieves significantly improved signal strength map prediction accuracy compared to various baseline methods.

## 1. Overview

We presented the design of GEO2SIGMAP, an efficient framework for high-fidelity RF signal mapping leveraging geographic databases and a novel cascaded U-Net model. We first developed an automated pipeline that efficiently generates 3D building and path gain maps via the integration of a suite of open-sourced tools, including OSM, Blender and Sionna. Then, the cascaded U-Net model pre-trained on synthetic datasets utilizes the building map and sparse SS map as input to predict the full SS map for the target (unseen) area. We extensively evaluated the performance of GEO2SIGMAP using large-scale field measurement collected using three UE types across six CBRS LTE cells deployed on the Duke University West Campus. Our results showed that GEO2SIGMAP achieves significantly improved RMSE of the SS map prediction compared to existing baseline methods.


For a full technical description on GEO2SIGMAP, please read our paper (Will release to arXiv in a few days):

> Y. Li, Z. Li, Z. Gao, T. Chen,  "GEO2SIGMAP: High-Fidelity RF Signal Mapping Using Geographic Databases," 

This is an active project, if you want to have a community discussion, please start a new discussion thread in the discussion tab, and we will get back to you as soon as possible.


## 2. Repo Structure

|  Source Files      |  Description                                                                                                             |
|  -----             |  -----                                                                                                                   |
|  `gen_data/`   |  This folder contains the code of our pipeline to generate signal coverage map based on realworld building infos. |
|  `data/`    |  This folder contains the sample data generate by our pipeline.                                                 |
|  `ml/`       |  This folder contains the code of cascade machine learning models.                                                           |

## 3. Installation

We provide a details document for a clean install and also a docker image for quick try.

Note: the docker way may suffer from performance loss due to xvfb virtual framebuffer.
### 3.1 Docker

Simply run:
```console
docker run --name g2s -it ffkshakf/geo2sigmap:latest bash
```

We only provide a amd64 arch docker image. If you run on a Apple Silicon or ARM64, you can add `--platform linux/amd64` to run it under emulation.
```console
docker run --name g2s --platform linux/amd64 -it ffkshakf/geo2sigmap:latest bash
```


### 3.2 Install from Scratch
Note: Following command tested on a clean install Ubuntu:22.04, you can follow the Blender offcial document about compiling Blender [here](https://wiki.blender.org/wiki/Building_Blender) if you would like to run on other OS/ARCH.

#### Install Initial Packages
```console
sudo apt update
sudo apt install python3 python3-pip git
```
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
perl -i -pe 'BEGIN{undef $/;} s/(patch -d \$_src -p1 < \$SCRIPT_DIR\/patches\/usd\.diff\n\s+fi\n\ncd \$_src)/$1\n\n    sed -i.bak  -e '\''s\/\.if !defined\.ARCH_OS_WINDOWS\./#if 0\/'\'' -e '\''s\/\.if defined\.ARCH_COMPILER_GCC\..*/#if 0\/'\'' -e '\''s\/defined\.ARCH_COMPILER_CLANG\.\/\/'\'' -e '\''s\/\.if defined\.ARCH_OS_LINUX\./#if 0\/'\'' -e '\''s\/\.if !defined\.ARCH_OS_LINUX\./#if 1\/'\'' pxr\/base\/arch\/mallocHook.cpp/' ./build_files/build_environment/install_deps.sh
```

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

```
make update
make release -j20 BUILD_CMAKE_ARGS="-U XXXXXXXXXXXXXXXX
```
#### Download Blender add-on & Apply Changes
There are two add-ons, [Blosm](https://prochitecture.gumroad.com/l/blender-osm) and [mitsuba-blender](https://github.com/mitsuba-renderer/). Download the zip file and place them in the root of this project `*/geo2sigmap/`.
```console 
unzip mitsuba-blender.zip
perl -i -pe 's/result = subprocess\.run\(\[sys\.executable, '-m', 'ensurepip'\], capture_output=True\)\n\s+return result\.returncode == 0/return True/' mitsuba-blender/__init__.py
zip -r -0 mitsuba-blender.zip mitsuba-blender
```

#### Install Sionna
Please follow the Sionna's official document [here](https://nvlabs.github.io/sionna/installation.html).

#### Install Pytorch

Please follow the Pytorch's official document [here](https://pytorch.org/get-started/locally/).

### 4. Useage
#### Generate 3D building meshs & 2D building height map
We start with select a top left GPS coordinate and a bottom right coordinate. Simply put these two number into the `.env` file, and also defined the lang-to-building threshold to filter out the open space area. 

Then simply run:
```console
python3 gen_data/Step1_OSM/OSM_from_generate_nc_dataset.py
```

Note: The public OSM sever have a query limitation around 2-10 query/second, so if you want to achieve a faster process speed, consider deploy a self host OSM server following the OSM offcial document [here](https://wiki.openstreetmap.org/wiki/Overpass_API/Installation). A reasonable speed of self hosted server would be around 100-200 query/second on a SSD computer.

#### Generate signal coverage map using Sionna
To use sionna generate signal coverage map, run xxxx. The sionna cofigue is defined in xxxx.

#### Train the model
To train our model, run xxxxx. 












## 5.License

Distributed under the APACHE LICENSE, VERSION 2.0
