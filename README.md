
# Futurama Character Image Classification with PySpark and Spark MLlib

This project uses **Apache Spark** and **MLlib** to classify characters from the TV series *Futurama* in images. It leverages **OpenCV** for feature extraction, creating a scalable solution for multi-label classification where each image can contain multiple characters (Fry, Leela, Bender).

## Project Structure

- **data/train_data.csv**: CSV file containing labeled data for training.
  - **Columns**: 
    - `file`: Image filename.
    - `isFry`, `isLeela`, `isBender`: Binary labels indicating if each character appears in the image.
- **data/train_img/**: Folder containing training images.
- **data/test_img/**: Folder containing test images without labels, used to evaluate the models.
- **submission/**: Directory where the classified test results are saved as a CSV file.

## Prerequisites

- **Python 3.x**
- **Apache Spark** with **PySpark**
- **OpenCV** for image processing
- **NumPy** for handling image data

### Install Dependencies

```bash
pip install pyspark opencv-python-headless numpy
```

> **Note**: If working in a non-graphical environment, use `opencv-python-headless` to avoid graphical dependencies.

## Project Workflow

### 1. Spark Session and Data Loading

The project begins by creating a Spark session and reading data from:
- **train_data.csv** for labeled data.
- **train_img/** for training images.
- **test_img/** for test images, which are unlabeled and used to make predictions.

### 2. Extract Filenames and Join with Labels

Each image’s filename is extracted from its path, allowing us to join the training images with the labels in `train_data.csv`.

### 3. Feature Extraction with Color Histogram

Using OpenCV’s color histogram, image features are extracted to create a representative feature vector. These features are later converted to a format compatible with Spark MLlib, allowing seamless integration into the classification models.

### 4. Convert Features to DenseVector Format

Spark MLlib requires features in `DenseVector` format. The extracted features are transformed to this format to ensure compatibility with MLlib’s classification models.

### 5. Training Binary Models for Multi-Label Classification

For each character (Fry, Leela, Bender), a binary Random Forest model is trained to predict if the character appears in the image. This approach enables multi-label classification by applying separate models for each character.

### 6. Model Evaluation and Prediction on Test Set

For each trained model, predictions are made on the test set, and accuracy is calculated. Predictions are stored in specific columns for each character, making it easy to interpret results for each individual.

### 7. Preparing and Saving the Results

The final predictions are saved in a CSV file within `submission/`, containing predicted labels for each character in each test image.

## Notes

- **Scalability**: Spark’s distributed architecture allows for efficient processing of large datasets, making it suitable for scaling this project to larger image datasets.
- **Optimization**: Parameters for the Random Forest classifier, such as the number of trees, can be adjusted to improve classification performance.

## License

This project is licensed under the MIT License.
