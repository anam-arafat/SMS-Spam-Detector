# Spam Detection using Machine Learning

## 📋 Project Overview
This project implements a comprehensive machine learning system to classify SMS messages as **spam** or **ham** (non-spam). Using Natural Language Processing (NLP) techniques and various machine learning algorithms, the system can accurately identify spam messages with high precision.

**🎥 Learning Resource**: This project was developed following the tutorial: [**Project 17. Spam Mail Prediction using Machine Learning with Python**](https://www.youtube.com/watch?v=rxkGItX5gGE)

## 📊 Dataset
- **Source**: [**SMS Spam Collection Dataset - Kaggle**](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size**: 5,572 SMS messages
- **Classes**: 
  - **Ham** (Non-spam): 4,825 messages
  - **Spam**: 747 messages
- **Format**: CSV file with message labels and text content

## 🚀 Features
- **Text Preprocessing**: Lowercasing, tokenization, stopword removal, and stemming
- **Feature Engineering**: Character count, word count, and sentence count analysis
- **Multiple ML Models**: Comparison of 7 different algorithms
- **TF-IDF Vectorization**: Advanced text feature extraction
- **Interactive Interface**: Real-time spam detection widget
- **Model Persistence**: Save and load trained models

## 🛠️ Technologies Used
- **Programming Language**: Python 3.7+
- **Libraries**:
  - Data Processing: `pandas`, `numpy`
  - NLP: `nltk`
  - Machine Learning: `scikit-learn`, `xgboost`
  - Visualization: `matplotlib`, `seaborn`
  - Interactive Widgets: `ipywidgets`
  - Word Cloud: `wordcloud`

## 📁 Project Structure
```
spam-detection-using-machine-learning/
│
├── spam_detection.ipynb          # Main Jupyter notebook
├── spam_detection.py             # Python file
├── vectorizer.pkl                # Saved TF-IDF vectorizer
├── model.pkl                     # Trained ML model
└── README.md                     # Project documentation
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or Google Colab

### Step-by-Step Installation

1. **Clone or download the project files**

2. **Install required packages**:
```bash
pip install kagglehub nltk wordcloud scikit-learn xgboost matplotlib seaborn pandas numpy ipywidgets
```

3. **Download NLTK resources**:
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
```

4. **Set up Kaggle authentication** (for dataset download):
   - Create a `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment variables
   - Or use the provided dataset file directly

## 🎯 Methodology

### 1. Data Preprocessing
- Removed duplicate messages
- Encoded target labels (ham: 0, spam: 1)
- Handled missing values and irrelevant columns

### 2. Feature Engineering
- **Character Count**: Length of each message
- **Word Count**: Number of words per message
- **Sentence Count**: Number of sentences per message
- **Text Transformation**: Cleaning and stemming

### 3. Text Processing Pipeline
1. **Lowercasing**: Convert all text to lowercase
2. **Tokenization**: Split text into individual words
3. **Special Character Removal**: Keep only alphabetic characters
4. **Stopword Removal**: Remove common English stopwords
5. **Stemming**: Reduce words to their root form using Porter Stemmer

### 4. Model Training & Evaluation
The following machine learning algorithms were compared:

| Model | Accuracy | Precision |
|-------|----------|-----------|
| Naive Bayes | ~97% | ~100% |
| Logistic Regression | ~96% | ~97% |
| Random Forest | ~97% | ~98% |
| SVM | ~97% | ~97% |
| XGBoost | ~97% | ~96% |
| Decision Tree | ~93% | ~82% |
| K-Neighbors | ~91% | ~100% |

**⭐ Best Model**: Naive Bayes (selected for deployment due to highest precision)

## 📈 Results

### Key Findings
- **Spam messages** tend to be longer with more characters and words
- **Common spam indicators**: words like "free", "win", "prize", "claim", "urgent"
- **TF-IDF vectorization** with 3000 features provided optimal performance
- **Naive Bayes** achieved the best precision score of nearly 100%

## 💻 Usage

### Interactive Prediction
Run the Jupyter notebook and use the interactive widget to:
1. Enter an SMS message in the text box
2. Click "Check for Spam"
3. View the classification result with visual indicators

### Code Example
```python
# Load the saved model
import pickle
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Predict a message
def predict_spam(message):
    transformed_text = transform_text(message)
    text_vector = tfidf.transform([transformed_text]).toarray()
    prediction = model.predict(text_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test the function
result = predict_spam("Congratulations! You've won a $1000 gift card!")
print(result)  # Output: Spam
```

## 🎮 Example Predictions

### Spam Messages (Correctly Identified)
- "Congratulations! You've won a $1000 Walmart gift card. Text YES to claim your prize!"
- "URGENT: Your bank account has been compromised. Click here to secure it now!"
- "FREE entry to win £5000 cash prize! Reply NOW to claim your reward!"

### Ham Messages (Correctly Identified)
- "Hey, are we still meeting for lunch tomorrow?"
- "Mom, I'll be home late today. Don't wait for me for dinner."
- "Can you pick up some milk on your way home?"

## 🔮 Future Improvements
- Implement deep learning models (LSTM, BERT)
- Add real-time SMS filtering capability
- Develop mobile application integration
- Expand to multi-language support
- Implement model explainability features
- Add ensemble methods for improved performance

## 👥 Contributors
- Developed as a machine learning project for spam detection
- Based on tutorial from [YouTube Video](https://www.youtube.com/watch?v=rxkGItX5gGE)

## 📄 License
This project uses the SMS Spam Collection Dataset which is publicly available on Kaggle for educational and research purposes.

## 🤝 Acknowledgments
- **Dataset Provider**: UCI Machine Learning Repository
- **Tutorial Source**: [Project 17. Spam Mail Prediction using Machine Learning with Python](https://www.youtube.com/watch?v=rxkGItX5gGE)
- **Libraries**: scikit-learn, nltk, pandas, and the open-source community

---

**⭐ Pro Tip**: The system works best with English SMS messages and may require retraining for other languages or modern spam patterns.