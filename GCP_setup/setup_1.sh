#!/bin/bash

sudo apt update

sudo apt upgrade

sudo apt-get update

sudo apt install wget

sudo apt install unzip

sudo apt install gcc

lspci | grep -i nvidia

uname -m && cat /etc/*release

gcc --version

uname -r

sudo apt-get install linux-headers-$(uname -r)

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin

sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1604-11-1-local_11.1.0-455.23.05-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604-11-1-local_11.1.0-455.23.05-1_amd64.deb

sudo apt-key add /var/cuda-repo-ubuntu1604-11-1-local/7fa2af80.pub

sudo apt-get update

sudo apt-get -y install cuda

export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64\ ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

sudo reboot
