### Team members:

- Vo Nhat Minh
- Tran Duc Duc
- Phuoc Ho
- Vladimir Nikolov

# IMPORTANT NOTE:

- All data file should be placed in a folder called "FC_TWENTE_FOLDER"
- All code file should be placed next to the "FC_TWENTE_FOLDER"
  Ex: Given current directory is "os/minh".
  Events data for game 1 will be in "os/minh/FC_TWENTE_FOLDER/Game 1/events.csv"
  File Project.ipynb will be in "os/minh/Project.ipynb"

# File structure:

## Project.ipynb

- The main file of project with the combination formula to compute Action Value using both Expected Threat and Pitch Control model

## data_in_out.py

- File for loading csv and preprocessing data (ex: change coordinate, group frame id, ...)

## ept.py

- File with methods for calculating Action Value

## pitchcontrol.py

- File with methods for calculating pitch control model

## processed_events.csv

- Generated file after calling preprocessing events cell in Project.ipynb

## velocities.py

- File with methods for calculating moving direction and speed of player

## visualization.py

- File with methods for plotting events and pitch control

## xT.csv

- Generated file after calling "xThreat_model.ipynb"

## xThreat_model.ipynb

- Expected Threat model. After running, it will generate xT.csv grid for expected threat values per position
