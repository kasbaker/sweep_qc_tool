# Data Management Classes
- ###`DataController`
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

- ###`FXData` - _`DataOperator`_
    - Inherits from _`DataOperator`_
    - Attributes:
        - `feature_data: JSONData` - raw feature extraction data
        - `fx_criteria : JSONData` - feature extraction criteria (future implementation)
    - Runs feature extraction from `nwb_data` based on good sweeps from `qc_data`

- ###`QCData` - _`DataOperator`_
    - Attributes:
        - `manual_qc_data : JSONData` - manual QC states selected by the user
        - `auto_qc_data : JSONData` - auto QC data from from running auto qc
        - `qc_criteria: JSONData` - criteria to use for auto QC
    - Runs auto QC based off of `qc_criteria`
    - Stores auto and manual QC states

- ###`NWBData` - _`DataContainer`_
    - Attributes:
        - `ontology_data : JSONData` - raw .json containing stimulus ontology
        - `stimulus_ontology : StimulusOntology` - stimulus ontology object used to create dataset
    - Operations:
        - `create_stimulus_ontology()` - creates stimulus ontology object from raw .json data
        - `create_ephys_dataset()` - creates ephys dataset using ontology and .nwb path

- ###`JSONData` - _`DataContainer`_
    - Attributes:
        - `display_action : QAction`
    - Owns `display_action : QAction` and `edit_action : QAction`
