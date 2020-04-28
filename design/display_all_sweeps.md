# Plan to display all sweeps

## Summary
Ephys would like to be able to display all of the sweeps in a particular .nwb file before they are filtered out by auto QC so they can perform manual QC on sweeps not included in the auto QC. Implementing this feature will require changes to data management architecture in order to keep track of the status of data storage and operations. Changes will also need to be made to the plotting classes in order to accommodate the display of voltage clamp sweeps.

## Goals
- Display all sweeps in the sweep table after an .nwb file is loaded
- Run auto QC by pressing a button or clicking a menu option
- Filter which sweeps are displayed by clicking a menu option

## Nice to have
- Progress bar for slow operations such as building the sweep table
- Filter sweeps by typing part of a stimulus code into a text input field
- Display initial test pulse for voltage clamp sweeps based on stimulus code

## Stepwise plan
1. Implement data storage classes, keeping auto qc pipeline and plotting as-is
2. Implement data operation classes, running auto qc and feature extraction from menu actions
3. Display all sweeps upon first loading a .nwb. Populate the auto-qc column upon running auto-qc
4. Implement options to filter which sweeps are displayed

## Classes to be implemented and what they do
See Attached UML diagrams for detailed architecture. Link here: 
[display_all_sweeps.pdf](https://github.com/AllenInstitute/sweep_qc_tool/files/4547602/display_all_sweeps.pdf)
### Data Management Classes
- ####`DataController` - `QWidget`
    - Owns high level data objects and connects them to GUI
    - Attributes: 
        - `nwb_data : NWBData` - holds raw ephys data and stimulus ontology
        - `fx_data : FXData` - runs feature extraction
        - `qc_data : QCData` - runs auto QC, holds manual and auto QC states
    - Operations:
        - `run_feature_extraction()` - sends `nwb_data` and `qc_data` to `fx_data.run()`
        - `run_auto_qc()` - sends `nwb_data` to `qc_data.run()`
        - `connect()` - connects the following attributes to other packages:
            - Raw data to sweep page and cell feature page
            - Actions and data states to menu bar
            - Status messages to main window

- ####`FXData` - _`DataOperator`_
    - Runs feature extraction from `nwb_data` based on good sweeps from `qc_data`
    - Attributes:
        - `feature_data: JSONData` - raw feature extraction data
        - `fx_criteria : JSONData` - feature extraction criteria (future implementation)

- ####`QCData` - _`DataOperator`_
    - Runs auto QC based off of `qc_criteria`
    - Stores auto and manual QC states
    - Attributes:
        - `manual_qc_data : JSONData` - manual QC states selected by the user
        - `auto_qc_data : JSONData` - auto QC data from from running auto qc
        - `qc_criteria: JSONData` - criteria to use for auto QC

- ####`NWBData` - _`DataContainer`_
    - Holds raw .nwb data and stimulus ontology
    - Attributes:
        - `ontology_data : JSONData` - raw .json containing stimulus ontology
        - `stimulus_ontology : StimulusOntology` - stimulus ontology object used to create dataset
    - Operations:
        - `create_stimulus_ontology()` - creates stimulus ontology object from raw .json data
        - `create_ephys_dataset()` - creates ephys dataset using ontology and .nwb path

- ####`JSONData` - _`DataContainer`_
    - Holds .json data
    - Attributes:
        - `display_action : QAction`
        - `edit_action : QAction`
    - Operations:
        - `display_json()` - displays a read-only output of the .json file
        - `edit_json()` - displays an editable output of the .json file
        
- ####_`DataOperator`_ - `QObject`
    - Abstract data operator class; `run()` implemented in child classes
    - Attributes:
        - `run_action : QAction` - menu action to trigger the data operation
        - `run_state : pyqtSignal` - status signal indicating state of data operation
    - Operations:
        - `run()` - performs the data operation (subclass implementation)

- ####_`DataContainer`_ - `QObject`
    - Abstract data container class; many attributes and operations implemented in child classes
    - Attributes:
        - `data_state : pyqtSignal` - indicates the state of the data (subclass implementation)
        - `status_mesage :pyqtSignal` - status message to send to the main window
        - `load_action : QAction` - menu action to trigger load dialog
        - `save_action : QAction` - triggers save dialog
        - `lims_download_action : QAction` - triggers download from lims (future implementation)
        - `lims_upload_action : QAction` - triggers upload from lims (future implementation)
        - `load_path : str` - location to load the data from
        - `save_path : str` - location to save the data
        - `data : Any` - holds the raw data (subclass implementation)
    - Operations:
        - `set_default_input()` - sets a file to load by default
        - `set_default_output()` - sets a file to save by default
        - `save()` - saves the data (subclass implementation)
        - `load()` - loads the data (subclass implementation)
        - `download_from_lims()` - downloads data from lims (future implementation in subclasses)
        - `upload_from_lims()` - uploads data to lims (future implementation in subclasses)    
        - `load_dialog()` - displays a dialog to load the data (subclass implementation)
        - `save_dialog()` - displays a dialog to save the data (subclass implementation)
        - `data_setter()` - sets the data (subclass implementation)
        - `data()` - returns the data
        
### Sweep Page Classes
More detailed plotting plan to come...
- #### `SweepPage` - `QWidget`
    - Widget containing sweep info and sweep plots
- #### `SweepTableModel` - `QAbstractTableModel`
    - Abstract container that holds data for the sweep page
- #### `SweepTableView` - `QTableView`
    - Interprets data from abstract table model and displays it to user
- #### `FixedPlots` - `NamedTuple`
    - Contains a thumbnail and popup plot of either test pulse or experiment
- #### `PulsePopup` - `PopupPlot`
    - Makes a popup graph of the test pulse when called
- #### `ExperimentPopup` - `PopupPlot`
    - Makes a popup graph of the experiment when called
- #### `PopupPlot` - `object`
    - Makes a generic ephys popup graph when called
- #### `SweepPlotter` - `object`
    - Generates thumbnails and popup plots to pass back to sweep table model