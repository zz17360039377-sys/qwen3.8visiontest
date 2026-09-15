#!/bin/bash
# Orchestrates all dataset jobs sequentially (server has a single slot).
cd /home/dreamchaser/selflabel/testenv/work
W=run_ds.py
python3 $W --dataset cls_armor_digit_slices --type cls --n 25 --workers 3 --max-tokens 60
python3 $W --dataset cls_armor_pattern_public --type cls --n 25 --workers 3 --max-tokens 60
python3 $W --dataset detect_rm2025_armor --type detect --n 30 --workers 3 --max-tokens 600
python3 $W --dataset detect_light_luminous --type detect --n 20 --workers 3 --max-tokens 600
python3 $W --dataset pose_okbuff2 --type pose --n 15 --workers 3 --max-tokens 900
python3 $W --dataset pose_droneok --type pose --n 15 --workers 3 --max-tokens 900
python3 $W --dataset labelme_radar_car --type labelme --n 12 --workers 3 --max-tokens 600
echo "ALL_DONE" > /home/dreamchaser/selflabel/testenv/work/logs/DONE.marker
