#!/bin/bash

wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh

bash Anaconda3-2021.11-Linux-x86_64.sh

source .bashrc

sudo apt-get install git

git clone github.com/orrzohar/PROB.git

sudo systemctl enable nvidia-persistenced

cd ..

cd ..

cd ..

sudo cp "/lib/udev/rules.d/40-vm-hotadd.rules" /etc/udev/rules.d

sudo sed -i '/SUBSYSTEM=="memory", ACTION=="add"/d' /etc/udev/rules.d/40-vm-hotadd.rules

sudo reboot
