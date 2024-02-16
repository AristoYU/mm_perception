#!/bin/bash
###
 # @Author       : yuhang09 yuhang09@baidu.com
 # @Date         : 2024-02-16 17:53:53
 # @LastEditors: AristoYU
 # @LastEditTime: 2024-02-17 02:27:57
 # @FilePath: /mm_perception/setup.bash
 # @Description  : 
 # Copyright (c) Baidu, Inc. and its affiliates. All Rights Reserved
### 

echo "###################################"
echo "# STEP 0: init deps               #"
echo "###################################"
git submodule update --init --recursive

# install mm deps
echo "###################################"
echo "# STEP 1: install mmlab deps      #"
echo "###################################"
# install mmcv
echo "###################################"
echo "# STEP 1.1: install mmcv          #"
echo "###################################"
cd ./modules/mmcv
pip3 install -r requirements/optional.txt -r requirements/runtime.txt
pip3 install -v -e .
cd -
# install mmengine
echo "###################################"
echo "# STEP 1.2: install mmengine      #"
echo "###################################"
cd ./modules/mmengine
pip3 install -r requirements/runtime.txt
pip3 install -v -e .
cd -
# install mmpretrain
echo "###################################"
echo "# STEP 1.3: install mmpretrain    #"
echo "###################################"
cd ./modules/mmpretrain
pip3 install -r requirements/runtime.txt
pip3 install -v -e .
cd -
# install mmdetection
echo "###################################"
echo "# STEP 1.4: install mmdetection   #"
echo "###################################"
cd ./modules/mmdetection
pip3 install -r requirements/optional.txt -r requirements/runtime.txt
pip3 install -v -e .
cd -
# install mmdetection3d
echo "###################################"
echo "# STEP 1.5: install mmdetection3d #"
echo "###################################"
cd ./modules/mmdetection3d
pip3 install -r requirements/optional.txt -r requirements/runtime.txt
pip3 install -v -e .
cd -
# install mmdeploy
echo "###################################"
echo "# STEP 1.6: install mmdeploy      #"
echo "###################################"
cd ./modules/mmdeploy
pip3 install -r requirements/optional.txt -r requirements/runtime.txt
pip3 install -v -e .
cd -