# SimpliQ-Question-Similarity-Predictor

This project involves building a machine learning model to predict whether two questions are similar, based on the Quora dataset. The model is developed using natural language processing (NLP) techniques and machine learning algorithms, with feature engineering and iterative improvements to achieve high accuracy. Additionally, a Streamlit app was created to provide a user-friendly interface for testing the model.

## Features

1. **Dataset**: Quora dataset containing pairs of questions labeled as similar or not similar.
2. **Feature Engineering**:
   - Basic features: Question length, word overlap, and more.
   - Bag of Words (BoW): Text vectorization using frequency counts.
   - Advanced NLP features: TF-IDF, semantic similarity, etc.
3. **Model**: Random Forest classifier used for training and prediction.
4. **Accuracy**: Iteratively improved by incorporating advanced features and optimizing the model.
5. **Streamlit App**: Interactive web application to test the model by inputting question pairs.

## Technologies Used

- **Programming Languages**: Python
- **Libraries and Frameworks**:
  - Machine Learning: Scikit-learn, TensorFlow
  - NLP: NLTK, SpaCy
  - Data Processing: Pandas, NumPy
  - Web App: Streamlit
- **Tools**: Jupyter Notebook, Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the app in your browser and input two questions to check their similarity.

## Model Workflow

1. **Preprocessing**:
   - Cleaned the dataset by removing stopwords, punctuation, and performing tokenization.
   - Normalized text using lemmatization.
2. **Feature Extraction**:
   - Engineered basic statistical features like word overlap and question length.
   - Utilized Bag of Words and advanced techniques like TF-IDF for text representation.
3. **Model Training**:
   - Trained a Random Forest classifier using the extracted features.
   - Tuned hyperparameters to optimize model performance.
4. **Evaluation**:
   - Evaluated model accuracy using metrics like precision, recall, F1-score, and confusion matrix.

## Streamlit App

The Streamlit app provides an intuitive interface where users can input two questions and check if they are similar. The app displays:
- The similarity prediction ("Similar" or "Not Similar").
- The model's confidence score for the prediction.

## Results

- Achieved significant accuracy improvements by iteratively refining features and leveraging the Random Forest algorithm.
- The Streamlit app allows seamless interaction for testing the model's performance.

## Future Work

- Explore deep learning models like BERT or transformers for better accuracy.
- Improve the app interface with additional visualization features.
- Add support for batch question similarity predictions.


## Acknowledgments

- Quora for providing the dataset.
- Open-source contributors of Scikit-learn, TensorFlow, Streamlit, and other libraries used in this project.

