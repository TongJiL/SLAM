# SLAM Algorithm

 
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
