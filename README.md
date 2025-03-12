## Overview
This project is an emotion detection system that classifies text into different emotions. It uses Natural Language Processing (NLP) techniques and a machine learning model to predict the emotion of a given sentence.

## Features
- **Preprocessing**: Cleans input text by removing stopwords and punctuation.
- **Training**: Uses a Naïve Bayes classifier to train on labeled text emotion data.
- **Prediction**: Predicts emotions for user input text.
- **Correction Mechanism**: Allows user feedback to improve predictions.
- **Data Management**: Merges existing datasets and corrected inputs for enhanced training.

## Installation
Ensure you have Python installed and run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

## Files Structure
- `preprocess.py` - Handles text cleaning and preprocessing.
- `train.py` - Trains the machine learning model.
- `predict.py` - Runs predictions on user input.
- `data.csv` - Contains the dataset for training.
- `emotiondata.csv` - Additional emotion dataset merged with `data.csv`.
- `model/` - Directory containing the trained model and vectorizer.

## Usage
### Training the Model
To train the model, run:
```bash
python train.py
```
This will process the dataset and train the model.

### Making Predictions
To predict the emotion of a sentence, run:
```bash
python predict.py
```
Enter a sentence when prompted, and the model will return a predicted emotion.

## Improving Accuracy
1. Add more labeled data to `data.csv`.
2. Retrain the model using `train.py` after updating the dataset.
3. Use the correction feature in `predict.py` to refine predictions.

## Notes
- Ensure `data.csv` is up-to-date before training.
- The trained model is saved in the `model/` directory.
- Visualization (`visualize.py`) has been removed as part of cleanup.

## License
This project is for educational purposes and is open-source.

