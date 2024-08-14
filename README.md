
# Domain Adaptation using GAN (DANN.py)

This repository contains the implementation of a Domain Adversarial Neural Network (DANN) for domain adaptation using Generative Adversarial Networks (GANs). The model is designed to minimize the discrepancy between the source and target domains, making it effective in cross-domain classification tasks.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Examples](#examples)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Domain Adaptation is a technique used to address the issue of domain shift, where the data distribution between the training (source) and testing (target) sets differs. This repository implements a DANN model that aligns the feature distributions between the source and target domains using adversarial training.

## Installation

### Prerequisites

- Python 3.6 or higher
- PyTorch 1.7 or higher
- Other dependencies listed in `requirements.txt`

### Clone the Repository

```bash
git clone https://github.com/moienmovafagh/Domain_adaptation_GAN.git
cd Domain_adaptation_GAN
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the DANN model on your dataset, run the following command:

```bash
python DANN.py --source_dataset path/to/source --target_dataset path/to/target --epochs 50 --batch_size 32
```

### Command Line Arguments

- `--source_dataset`: Path to the source domain dataset.
- `--target_dataset`: Path to the target domain dataset.
- `--epochs`: Number of epochs to train the model.
- `--batch_size`: Batch size for training.

### Example

```bash
python DANN.py --source_dataset ./data/source --target_dataset ./data/target --epochs 50 --batch_size 32
```

## File Structure

- `DANN.py`: Main script implementing the DANN model.
- `data/`: Directory for storing source and target datasets.
- `models/`: Directory for saving trained models.
- `results/`: Directory for saving training results and logs.
- `requirements.txt`: Python dependencies for the project.

## Examples

Below are some example use cases of the DANN model:

- **Cross-Domain Classification:** Train the model on digit datasets like MNIST (source) and SVHN (target) to classify digits across different domains.
- **Object Recognition:** Apply domain adaptation to object recognition tasks where the source domain contains labeled images and the target domain contains unlabeled images from a different distribution.

## Results

The trained DANN model reduces the domain discrepancy, achieving better generalization on the target domain. Below are some example results:

- **Accuracy on Source Domain:** 95%
- **Accuracy on Target Domain:** 85% (with domain adaptation)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute to this project.

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please contact [Moien Movafagh](moienmovafagh@gmail.com).
