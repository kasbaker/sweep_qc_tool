# Restructuring of data classes

- `data_monitor: DataMonitor`
    - Replaces `pre_fx_controller`
    - Connected to `data_mediator`,`data_manager`, `fx_data`, and `qc_data`
    - Monitors the data to see if it has changed; emits signals as necessary
  
- `data_mediator: DataMediator`
    - Takes over load and save dialogs handled by `pre_fx_controller`

- `data_manager: DataManager`
    - Replaces `pre_fx_data: PreFxData`
    - Handles loading and saving of data, creates data set from .nwb
    - Owns the raw data

- `fx_data: FxData`
    - Runs feature extraction on the data set
    - Holds feature extracted data
    
- `qc_data: QCData` 
    - Keeps track of sweep QC states
    - Runs auto qc and calculates sweep qc features






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