# Restructuring of data classes

- `data_controller : DataController`
    - Attributes: 
        - `nwb_data : NWBData`
        - `fx_data : FXData`
        - `qc_data : QCData`
    - Operations:
        - `connect()` - connects the following attributes to other packages:
            - Raw data to sweep page and cell feature page
            - Actions and data states to menu bar
            - Status messages to main window
        - `run_feature_extraction()` - sends `nwb_data` and `qc_data` to `fx_data.run()`
        - `run_auto_qc()` - sends `nwb_data` to `qc_data.run()`

- `FXData`
    - Inherits from _`DataOperator`_
    - Attributes:
        - `feature_data: JSONData` - raw feature extraction data
        - `nwb_data : NWBData` - raw ephys data to extract features from
        - `qc_data : QCData` - QC states with good sweeps to use for feature extraction
        - `fx_criteria : JSONData` - feature extraction criteria (future implementation)
    - Runs feature extraction from `nwb_data` based on good sweeps from `qc_data`

- `QCData`
    - Inherits from _`DataOperator`_
    - Attributes:
        - `manual_qc_data : JSONData` - manual QC states selected by the user
        - `auto_qc_data : JSONData` - auto QC data from from running auto qc
        - `qc_criteria: JSONData`
    - Runs auto QC based off of `qc_criteria`
    - Stores auto and manual QC states

- `NWBData`
    - Inherits from _`DataContainer`_
    - Owns `stimulus_ontology : JSONData`
    - Creates a stimulus ontology object from the raw json data
    - Holds raw data from .nwb files

- `JSONData`
    - Inherits from _`DataContainer`_
    - Owns `display_action : QAction` and `edit_action : QAction`

- `fx_data: FxData`
    - Runs feature extraction on the data set
    - Holds feature extracted data
    
- `qc_data: QCData` 
    - Keeps track of sweep QC states
    - Runs auto qc and calculates sweep qc features

- fx_data.run_feature_extraction.extract_data_set_features(data_set)
    - returns: cell_features, sweep_features, cell_record, sweep_records
    

    
    
 




- `settings: Settings`
    - Holds contextual information, like lims credentials ipfx version, and nwb paths. 
    - Contains methods called back from menu actions, (e.g.a dialog for loading a new file
    - Emits signals to `pre_fx_data` or `fx_data` when relevant settings change (e.g. if a new file is loaded, `pre_fx_data` becomes invalid).
- `pre_fx_data : PreFxData`
    - Data and workflow container for all pre-fx data
    - User data modifications from e.g. menu items (load a new nwb file) or sweep qc actions modify data here. 
    - Calculates and holds raw and derived data from sweep extraction and auto qc stages.
- `fx_data: FxData` 
    - Calculates and holds feature extraction results or unextracted state.
    - Extraction triggered manually via menu action.