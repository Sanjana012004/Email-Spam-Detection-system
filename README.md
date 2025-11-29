# 📧 Spam Email Detection System

A machine learning-powered web application that detects spam messages using Natural Language Processing (NLP) and Naive Bayes classification. Built with Streamlit for an interactive and beautiful user interface.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)

## 🌟 Features

- **Real-time Spam Detection**: Instantly classify messages as spam or safe
- **Machine Learning Model**: Uses Multinomial Naive Bayes classifier
- **High Accuracy**: Displays both accuracy and precision metrics
- **Beautiful UI**: Modern gradient design with smooth animations
- **Interactive Interface**: Easy-to-use text input with instant validation
- **Performance Metrics**: Real-time display of model accuracy and precision
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📊 Model Performance

The spam detection model is trained using:
- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: CountVectorizer with English stop words removal
- **Train/Test Split**: 80/20 ratio with random_state=42 for reproducibility
- **Metrics Tracked**: 
  - Accuracy: Overall correctness of predictions
  - Precision: Accuracy of spam predictions (reduces false positives)

## 🚀 Demo

The application features:
- Purple gradient background (elegant design)
- Animated result cards (red for spam, green for safe)
- Real-time performance metrics display
- Smooth hover effects and transitions

## 📁 Project Structure

```
spam-detector/
│
├── SpamDetection.py    # Main Streamlit application (your provided code)
├── spam.csv            # Training dataset (required)
└── requirements.txt    # Python dependencies
```

**Note**: This is a minimal setup with 3 files - the application, dataset, and dependencies file.

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Download the Files**
   - Save the Python code as `SpamDetection.py`
   - Save the requirements file as `requirements.txt`
   - Ensure `spam.csv` is in the same directory

2. **Install Required Libraries**

Using requirements.txt (recommended):
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas scikit-learn
```

3. **Verify Files**

Make sure you have all three files in the same folder:
```
your-folder/
├── SpamDetection.py
├── spam.csv
└── requirements.txt
```

4. **Run the Application**

```bash
streamlit run SpamDetection.py
```

The application will open in your default browser at `http://localhost:8501`

## 📦 Dependencies

The application requires 3 Python libraries specified in `requirements.txt`:

```txt
streamlit>=1.0.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

**Install all dependencies at once:**
```bash
pip install -r requirements.txt
```

**What each library does:**
- `streamlit` - Web interface framework for creating the interactive UI
- `pandas` - Data manipulation and CSV file reading
- `scikit-learn` - Machine learning library (includes CountVectorizer, MultinomialNB, train_test_split, precision_score)

## 💻 Usage

### Basic Usage

1. **Launch the Application**
   ```bash
   streamlit run SpamDetection.py
   ```

2. **Enter a Message**
   - Type or paste any message into the input box
   - Example: "Congratulations! You've won a free iPhone!"

3. **Click Validate**
   - The system will analyze the message
   - Results display instantly with color-coded indicators:
     - 🚨 **Red**: Spam detected
     - ✅ **Green**: Safe message

4. **View Performance Metrics**
   - Accuracy and Precision scores displayed below results
   - Metrics show how well the model performs

### Example Messages to Test

**Spam Messages:**
```
- "URGENT! You have won $1,000,000. Claim now!"
- "Click here for free iPhone. Limited time offer!"
- "Congratulations! You've been selected as a winner"
```

**Safe Messages:**
```
- "Hey, are you free for coffee tomorrow?"
- "Meeting scheduled for 3 PM today"
- "Can you send me the report by EOD?"
```

## 🧠 How It Works

### 1. Data Preprocessing
- Loads spam dataset from CSV file
- Removes duplicate entries
- Replaces labels: 'ham' → 'Not Spam', 'spam' → 'Spam'

### 2. Feature Extraction
- Uses **CountVectorizer** to convert text to numerical features
- Removes English stop words (common words like 'the', 'is', 'at')
- Creates a bag-of-words representation

### 3. Model Training
- Splits data: 80% training, 20% testing
- Trains **Multinomial Naive Bayes** classifier
- Naive Bayes is ideal for text classification due to its efficiency

### 4. Prediction
- User input is vectorized using the same CountVectorizer
- Model predicts spam probability
- Returns classification: Spam or Not Spam

### 5. Performance Evaluation
- **Accuracy**: Percentage of correct predictions (both spam and ham)
- **Precision**: Of all messages predicted as spam, how many are actually spam
  - High precision = fewer false positives (safe messages marked as spam)

## 🎨 UI Customization

### Changing Colors

Edit the CSS in the `st.markdown()` section:

```python
# Main gradient background
.stApp {
    background: linear-gradient(135deg, #your-color-1 0%, #your-color-2 100%);
}

# Spam result color
.result-spam {
    background: linear-gradient(135deg, #your-red-1 0%, #your-red-2 100%);
}

# Safe result color
.result-safe {
    background: linear-gradient(135deg, #your-green-1 0%, #your-green-2 100%);
}
```

### Modifying Layout

Adjust Streamlit configuration:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="Your Icon",
    layout="centered"  # or "wide"
)
```

## 📈 Model Performance Tips

### Improving Accuracy

1. **Larger Dataset**: Use more training examples
2. **Feature Engineering**: 
   - Add TF-IDF instead of CountVectorizer
   - Include bigrams/trigrams
3. **Hyperparameter Tuning**: Adjust alpha parameter in MultinomialNB
4. **Try Different Models**: 
   - Logistic Regression
   - Support Vector Machines
   - Random Forest

### Example: Using TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Replace CountVectorizer with TfidfVectorizer
cv = TfidfVectorizer(stop_words='english', max_features=5000)
```

## 🔧 Troubleshooting

### Common Issues

**Issue 1: "spam.csv not found"**
```
Solution: Ensure spam.csv is in the same directory as app.py
```

**Issue 2: Module not found errors**
```bash
Solution: Install missing packages
pip install streamlit pandas scikit-learn
```

**Issue 3: Low accuracy (<80%)**
```
Solution: 
- Check dataset quality
- Ensure balanced classes (similar number of spam and ham)
- Add more training data
```

**Issue 4: Streamlit won't start**
```bash
Solution: Check if port 8501 is available
streamlit run SpamDetection.py --server.port 8502
```

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Create a GitHub repository
2. Upload all three files to the repository:
   - `SpamDetection.py`
   - `spam.csv`
   - `requirements.txt`
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub account
5. Select your repository and specify `SpamDetection.py` as the main file
6. Streamlit Cloud will automatically detect and install dependencies from `requirements.txt`
7. Click "Deploy"

### Local Sharing

To share with others locally:
1. Share all three files:
   - `SpamDetection.py`
   - `spam.csv`
   - `requirements.txt`
2. Ask them to run:
   ```bash
   pip install -r requirements.txt
   streamlit run SpamDetection.py
   ```

## 📊 Dataset Information

### Your spam.csv File

Your dataset should have this structure:

| Column | Description |
|--------|-------------|
| Category | Label: 'ham' (not spam) or 'spam' |
| Message | The text content of the message |

Example format:
```csv
Category,Message
ham,Hey how are you doing today?
spam,WINNER! You have won a $1000 prize. Click here now!
ham,Can we meet for lunch tomorrow?
spam,Congratulations! You've been selected for a free vacation...
```

### Sample Dataset Sources

If you need a spam dataset, download from:
- [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- [Enron Email Dataset](https://www.kaggle.com/wcukierski/enron-email-dataset)
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)

**Important**: Save the downloaded dataset as `spam.csv` in the same folder as `SpamDetection.py`

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created with ❤️ using Streamlit and scikit-learn

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web framework
- **scikit-learn** - For machine learning tools
- **UCI Machine Learning Repository** - For spam datasets

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please give it a star!**
