#!/bin/bash

GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/EVAL_M_OWOD_BENCHMARK.sh
