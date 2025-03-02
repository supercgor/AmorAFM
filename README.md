# Unveiling the amorphous ice layer during ice premelting using AFM integrating machine learning

by
Binze Tang†, 
Chon-Hei Lo†,
Tiancheng Liang†,
Jiani Hong†,
Mian Qin,
Yizhi Song,
Duanyun Cao,
Ying Jiang*, 
Limei Xu*,

> Machine learning framework for reconstructing bulk ice surface structure from AFM images.

<!-- ![](manuscript/figures/hawaii-trend.png) -->

<!-- *Caption for the example figure with the main results.* -->


## Software implementation

> Briefly describe the software that was written to produce the results of this
> paper.

This repository contains code for a novel ML-AFM framework for 3D atomic reconstruction of disordered interfaces, exemplified by ice surfaces, directly from Atomic Force Microscopy (AFM) data. To address the challenges of AFM's limited depth sensitivity and signal complexity in characterizing 3D disordered interfacial structures, our framework uniquely integrates three neural networks:

CycleGAN (Noise Augmentation): Enhances robustness by learning realistic noise from experimental AFM images and augmenting simulated training data.
3D U-Net-like Object Detection Network: Precisely identifies the top-layer atomic structure from AFM signals.
Conditional Variational Autoencoder (cVAE) Structure Generation Network: Infers the complete 3D structure by generating subsurface layers conditioned on the detected top-layer.

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

And download the data and pre-trained models from the following link to `dataset` folder:
    ...

To predict the AFM images, you should prepare directories as follows:

- 📦 ss0
-  ┣ 📜 0.png
-  ┣ 📜 1.png
-  ┣ 📜 2.png
-  ┣ 📜 ...
-  ┣ 📜 cell.txt

AFM Images: Saved as .png files and named sequentially in ascending order of tip-sample distance.
cell.txt: This file contains a comment line followed by a line with two numbers specifying the width and height of the AFM images in nm.

Run the code:

    python3 tools/eval.py -i dataset/ss

Running the Object Detection Network on CPU:

    python3 tools/eval.py -i dataset/ss --detect-only

This will let the network predict the topmost layer structure based on the input AFM images located in dataset/ss0. The predicted structure will be saved in .xyz format. 
Before proceeding to generate the complete 3D structure, you can manually modify the predicted topmost layer structure (e.g., using structure editing software), or adjust the hydrogen orientation of the topmost layer through energy minimization techniques.
Refer to the paper for detailed guidance on these optional steps and their potential benefits.

Running the Generation Network on CPU:

    python3 tools/eval.py -i dataset/ss.xyz --match-only

With the topmost layer structure already predicted (or provided), this command will trigger the generation network to reconstruct the complete 3D structure. The network utilizes the topmost layer structure as a conditional input to infer the underlying layers.
After generating the complete 3D structure, you can proceed further simulations and analyses as detailed in the paper's instructions.

To get help on the command line options, run:

    python3 tools/eval.py --help


## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

<!-- The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
JOURNAL NAME. -->
