# SLAM Algorithm
Simultaneous localization and mapping (SLAM) requires the robot to start from an unknown location in a completely unknown environment, and use the sensor to observe the environment to incrementally establish the navigation map of the environment. At the same time, according to the established map synchronization to determine their position, and thus comprehensively answer the question ”Where am I?” and finally evaluate the surrounding environment.

This poject presented approaches using SLAM system to solve the localization problem of a differential-drive robot and using depth and RGB camera to texture the map. This work shows the detailed steps of using particle filter method to localize the accurate location of the robot and bulit the map detected by the robot. Finally texture the map with real color segment.

![image](https://github.com/TongJiL/SLAM/blob/master/image/20.jpeg)

Description
===========
This repo provides some implements of the SLAM algorithm on an in-door moving robot.

Code organization
=================
load_data.py    --Provide the way to load date from sensors

map_utils.py    --The method to build grid map

TEST.py         --The method to build the map using dead-reckoning

particles.py    --The method to use particle filter to do the SLAM

texture.py      --Texturing the map to put the coresponding color on the map after SLAM

Runs
==========
Each code can be run directly. However, it would take some time to get results from particles.py and texture.py(Approximately 2 hours).

Results
============
rep.pdf contains the summary of the method and the results.
