# Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases
[![arXiv](https://img.shields.io/badge/arXiv-2312.14303-green?color=FF8000?color=009922)](https://arxiv.org/abs/2312.14303)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/dvlab-research/LongLoRA/blob/main/LICENSE)


Welcome to the Geo2SigMap. Here are the documents for our automated framework. 

## TABLE OF CONTENTS
1. [Overview](#overview)
2. [Folder Structure](#folder-structure)
3. [Installation](#installation)
4. [Example Usage](#example-usage)
5. [License](#license)

## Overview

* **We Designs an automated framework that integrates open-source tools, including geographic databases (OSM), computer graphics (Blender), and ray tracing (Sionna), and supports scalable ray tracing and RF signal mapping at-scale using real-world building information;**

## Folder Structure
   
## Installation

We provide detailed guidelines for installation including a docker image (for quick try) and a clean install.



### Docker
>Note: Docker may suffer from performance loss due to xvfb virtual framebuffer.

Run the following command to use our pre-compiled docker image:
```console
docker run --name g2s -it ffkshakf/geo2sigmap:latest bash
```

We only provide an amd64 arch docker image. If you run on a Apple Silicon or ARM64, you can add `--platform linux/amd64` to run it under emulation.
```console
docker run --name g2s --platform linux/amd64 -it ffkshakf/geo2sigmap:latest bash
```

### Install from Scratch
Note: The following commands have been tested on a clean install Ubuntu:22.04, you can follow the Blender offcial document about compiling Blender [here](https://wiki.blender.org/wiki/Building_Blender) if you would like to run on other OS/ARCH.

#### Install Initial Packages
```console
sudo apt update
sudo apt install python3 python3-pip git
pip install --target=/usr/lib/python3/dist-packages Cython
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
perl -i -pe 'BEGIN{undef $/;} s/(patch -d \$_src -p1 < \$SCRIPT_DIR\/patches\/usd\.diff\n\s+fi\n\n\s+cd \$_src)/$1\n\n    sed -i.bak  -e '\''s\/\.if !defined\.ARCH_OS_WINDOWS\.\/#if 0\/'\'' -e '\''s\/\.if defined\.ARCH_COMPILER_GCC\..*\/#if 0\/'\'' -e '\''s\/defined\.ARCH_COMPILER_CLANG\.\/\/'\'' -e '\''s\/\.if defined\.ARCH_OS_LINUX\.\/#if 0\/'\'' -e '\''s\/\.if !defined\.ARCH_OS_LINUX\.\/#if 1\/'\'' pxr\/base\/arch\/mallocHook.cpp/' ./build_files/build_environment/install_deps.sh
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

```console 
make update
make release -j20 BUILD_CMAKE_ARGS="-U XXXXXXXXXXXXXXXX
```

#### Download Blender Add-on & Apply Changes
There are two add-ons, [Blosm](https://prochitecture.gumroad.com/l/blender-osm) and mitsuba-blender( Already included in this project ). Download the zip file and place them in the folder `*/geo2sigmap/gen_data`.
<!-- ```console 
unzip mitsuba-blender.zip
perl -i -pe 's/  result = subprocess\.run\(\[sys\.executable, '-m', 'ensurepip'\], capture_output=True\)\n\s+return result\.returncode == 0/return True/' mitsuba-blender/__init__.py
zip -r -0 mitsuba-blender.zip mitsuba-blender
``` -->

#### Install Sionna
Please follow [Sionna's official document](https://nvlabs.github.io/sionna/installation.html) to install Sionna.

>Note: Our code is tested on Sionna v0.15.0

## Example Usage
>Note: The public OSM sever has a query limitation of around 2-20 queries/second. To achieve higher speed, consider deploying a self-host OSM server following the [OSM official document](https://wiki.openstreetmap.org/wiki/Overpass_API/Installation). A reasonable speed of a self-hosted server would be above 200 queries/second.

The following command uses an 8km^2 region in Durham, NC, 512m area dimension and 0.2 building to landing threshold as an example. For custom settings, please check the detail option by the `-h` arguments for each Python script.
![image](https://github.com/functions-lab/geo2sigmap/assets/24806755/02315890-e317-4232-b98f-015e84a16118)

### Example structure of result data folder
    .
    ├── data
    │   ├── measurement
    │   ├── generated
    │   │   ├── Jan05_1837_01b9be                                                 # Step 1: Each folder for a given region
    │   │   │   ├── Area_b2l_result.txt                                           #         Each line recorded the area's GPS coordinate and b2l ratio
    │   │   │   ├── Filtered_Area_b2l_result.txt                                  #         Same as above but filtered with bl2 ratio > THRESHOLD
    │   │   │   ├── OSM_download                                                  #         OSM XML file,  each .xml file for a given area
    │   │   │   │   ├── 77ecdcb3-f018-4606-81f9-5694be5ec41d.xml                     
    │   │   │   │   └── ...
    │   │   │   ├── Bl_building_npy                                               # Step 2: 2D building height maps, each .npy file for a given area
    │   │   │   │   ├── 77ecdcb3-f018-4606-81f9-5694be5ec41d.npy                      
    │   │   │   │   └── ...
    │   │   │   ├── Bl_xml_files                                                  # Step 2: 3D Building Meshs
    │   │   │   │   ├── 77ecdcb3-f018-4606-81f9-5694be5ec41d                      #         Each folder for a given area   
    │   │   │   │   │   ├── meshes
    │   │   │   │   │   │   ├── xxxx.ply                                          #         Details mesh info about each building inside this area
    │   │   │   │   │   │   └── ...
    │   │   │   │   │   ├── 77ecdcb3-f018-4606-81f9-5694be5ec41d.xml              #         Manifest for all buildings' mesh info
    │   │   │   │   │   └── ...
    │   │   │   │   └── ...
    │   │   │   └── Jan05_1837_RXcross_TXtr38901-cross_SampleNum7e6_cmres4        # Step 3: Signal coverage map generated by Sioona RT, 
    │   │   │       │                                                                            each folder for a given TX and RX settings.
    │   │   │       ├── 77ecdcb3-f018-4606-81f9-5694be5ec41d_-244_-244_21_82.npy  #         Each file for a given TX position and angle.
    │   │   │       └── ...
    │   │   └── ...

    


### Step 1. Pre-check the building-to-land ratio and download the OSM XML file for the target area
```console
python3 gen_data/Step1_OSM/OSM_PreCheck_and_Download.py
```

The output of this step should look like this(min_lon, max_lat, max_lon, min_lat, b2l_ratio, uuid ):
> -78.931571,36.017079,-78.921207,36.007433,0.767999,12862155-54ef-44ac-95cf-7aa7d0a862d4

### Step 2. Generate 3D Building Meshs & 2D Building Height Map

```console
python3 gen_data/Step2_Blender/blender_wrapper.py --data-dir=[Data dir generated by first step.]
```



### Step 3. Generate synthetic signal coverage map by Sionna RT
```console
python3 gen_data/Step3_Sionna/xml_to_heatmap_wrapper512_tr38901_randAngle.py --data-dir=[Data dir generated by first step.]
```
The above command has a default setting of ISO antenna.
```console
python3 gen_data/Step3_Sionna/xml_to_heatmap_wrapper512_tr38901_randAngle.py --data-dir=[Data dir generated by first step.]
```
The above command has a default setting of TR38901 directional antenna with four random select angles for each area.

An example of Step 3 output:

![An demo of Step 3](https://github.com/functions-lab/geo2sigmap/assets/24806755/e74edf95-1299-428c-bdad-d86244f1a90c)




<!--- #### Generate signal coverage map using Sionna
To use sionna generate signal coverage map, run xxxx. The sionna cofigue is defined in xxxx.

#### Train the model
To train our model, run xxxxx. ---> 

## License

Distributed under the APACHE LICENSE, VERSION 2.0