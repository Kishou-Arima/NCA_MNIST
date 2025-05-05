# Neural Cellular Automata for Digit Generation

This repository contains an implementation of an advanced Neural Cellular Automata (NCA) model developed using PyTorch for the purpose of generating handwritten digit images from the MNIST dataset. The model integrates digit-specific embeddings that significantly enhance digit-awareness and improve the quality of image reconstruction.

## Overview

Neural Cellular Automata (NCA) are computational models inspired by biological systems that evolve grid-based cellular states through localized interactions. This implementation specifically focuses on generating and accurately reconstructing images of handwritten digits, employing label embeddings to guide and refine the generative process.

## Features

* **Digit-Specific Label Embeddings**: Enhances the NCA's digit-awareness by incorporating embeddings tailored for each digit class.
* **Pre-trained Initialization**: Model initialization with weights pre-trained on individual digit classes, further fine-tuned into a unified model.
* **Dynamic Training Routine**: Utilizes intermediate step-based loss computation to progressively refine generated images.
* **Comprehensive Evaluation**: Includes evaluation procedures with pixel accuracy metrics to quantitatively assess model performance.
* **Visualization Tools**: Provides scripts for visual comparisons between original and NCA-generated digits.

## Installation

### Requirements

* Python 3.x
* PyTorch
* Torchvision
* NumPy
* Matplotlib

Install dependencies using:

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

### Training

To fine-tune the unified NCA model:

```bash
python your_script_name.py
```

### Evaluation

After training, evaluate and visualize model performance:

```bash
python your_script_name.py
```

## Results

The trained NCA model effectively generates accurate and visually coherent digit images, demonstrating competitive pixel accuracy on the MNIST dataset.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for any enhancements or improvements.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
