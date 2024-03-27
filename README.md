# Simple Neural Network for Cat vs Dog Classification

This script is designed to train a simple neural network to classify images as either cats or dogs using a dataset of images. It utilizes TensorFlow and Keras for model creation and training.

## Usage

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- NumPy
- matplotlib
- imutils
- OpenCV

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Sebastian352/Simple_Binary_NN
   ```

2. Navigate to the directory:

   ```bash
   cd repository
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Script

Use the following command to run the script:

```bash
python script.py --dataset path/to/dataset --model path/to/model --label-bin path/to/label/binarizer --plot path/to/plot
```

Replace `path/to/dataset`, `path/to/model`, `path/to/label/binarizer`, and `path/to/plot` with relevant file paths.

### Arguments

- `-d`, `--dataset`: Path to the input dataset of images (required).
- `-m`, `--model`: Path to the output trained model (required).
- `-l`, `--label-bin`: Path to the output label binarizer (required).
- `-p`, `--plot`: Path to the output accuracy/loss plot (required).

## Dataset

You can download a relevant dataset from [this link](https://www.kaggle.com/datasets/tongpython/cat-and-dog).

## Script Details

This script performs the following steps:

1. Loads the images from the dataset.
2. Preprocesses the images and labels.
3. Splits the dataset into training and testing sets.
4. Creates a simple neural network model using Keras.
5. Trains the model on the training data.
6. Evaluates the trained model on the testing data.
7. Generates a plot showing training and validation loss/accuracy.
8. Saves the trained model, label binarizer, and plot to specified paths.

## Acknowledgements

- This script is adapted from a tutorial on [Kaggle](https://www.kaggle.com/).
- The dataset used for training the model can be found on Kaggle.

## License

[MIT License](LICENSE)
