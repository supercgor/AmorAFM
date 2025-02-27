# AmorAFM


# Paper template for the pinga-lab (*replace with paper title*)

by
Binze Tang†, 
Chon-Hei Lo†,
Tiancheng Liang†,
Jiani Hong†,
Ying Jiang*, 
Limei Xu*,
et. al.

> Machine learning Framework for Amorphous Ice Layer Detection on AFM images.

<!-- ![](manuscript/figures/hawaii-trend.png) -->

<!-- *Caption for the example figure with the main results.* -->

## Abstract

Ice premelting plays a key role in atmospheric and biological processes but remains
poorly understood at the atomic level due to surface characterization limitations. We
report the discovery of a novel amorphous ice layer (AIL) preceding the quasi-liquid
layer (QLL), enabled by a machine learning framework integrating atomic force
microscopy (AFM) with molecular dynamics simulations. This approach overcomes
AFM's depth and signal limitations, allowing for three-dimensional surface structure
reconstruction from AFM images. We identify the AIL, present between 121-180K,
displaying disordered two-dimensional hydrogen-bond network with solid-like
dynamics, thereby refining the phase diagram of ice premelting. These results challenge
the conventional view that significant hydrogen-bond disorder is exclusive to the QLL
and offer new insights into surface growth dynamics, advancing AFM for three-
dimensional disordered interface studies.


## Software implementation

> Briefly describe the software that was written to produce the results of this
> paper.

This repository is maintained by Chon-Hei Lo. 
To train a new model, run the code in `src/train_*.py`
To predict the AFM images, run the code in `tools/eval.py`
Please put all the parameters and dataset folders (or soft links) in the `dataset` folder.
The output of all codes are by default saved in the `outputs` folder.
Modify the `configs/*.py` if necessary.

## Getting the code

You can download a copy of all the files in this repository by cloning the this repository:

    git clone https://github.com/supercgor/AmorAFM.git

<!-- or [download a zip archive](https://github.com/pinga-lab/PAPER-REPO/archive/master.zip). -->

<!-- A copy of the repository is also archived at *insert DOI here* -->

## Dependencies

The code is tested in Python 3.10, and developed in Ubuntu-24.04.

The following Python packages are required to run the code (see `requirements.txt`):
- ase
- matplotlib
- networkx
- numpy
- pandas
- pillow
- scikit-learn
- scipy
- torch
- torch-fidelity
- torchaudio
- torchmetrics
- torchvision
- tqdm

We recommend using the `virtualenv` package to manage Python environments.
To create a new environment with the required dependencies, run:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
   
## Usage
Before running any code you must activate the virtual environment:

    source .venv/bin/activate

And download the data and pre-trained models from the following link:
    ...

To predict the AFM images, you should prepare directories as follows:

- 📦 name_of_dir
-  ┣ 📜 0.png
-  ┣ 📜 1.png
-  ┣ 📜 2.png
-  ┣ 📜 ...
-  ┣ 📜 cell.txt

Run the code:

    python3 tools/eval.py -i name_of_dir

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

<!-- The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
JOURNAL NAME. -->