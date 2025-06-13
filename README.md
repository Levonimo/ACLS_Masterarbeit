# WARP

WARP (Warping Algorithm for Retention Profiles) is a graphical application that helps to preprocess and align TD-CIS-GC-MS chromatograms. It converts vendor files to mzML, performs baseline correction, aligns chromatograms by correlation optimized warping and provides basic PCA and statistical tools.

## Installation

Create the Conda environment or install the dependencies with pip.

### Conda
```bash
conda env create -f environment.yml
conda activate warp-env
```

### Pip
```bash
pip install -r requirements.txt
```

## Usage

Run the main GUI application:

```bash
python WARP.py
```

This opens the PyQt5 interface that guides you through data conversion, warping and analysis. Make sure `msconvert` from ProteoWizard is available in your `PATH`.

## Repository Structure

- `WARP.py` – entry point launching the GUI
- `component/` – GUI modules, data preparation and algorithms
- `Plots/` – supplementary scripts for data analysis and plotting
- `test_*.py` – example scripts for manual testing

## License

GPL3 / CC BY-NC-SA 4.0 as stated in the source files.

