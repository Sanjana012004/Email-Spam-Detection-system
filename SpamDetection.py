from sklearn.metrics import precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Page configuration
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 1rem 0;
    }
    
    h2 {
        color: white;
        text-align: center;
        font-size: 1.3rem;
        opacity: 0.95;
    }
    
    h3 {
        color: white;
        font-weight: bold;
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        background-color: white;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 12px;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 10px rgba(118, 75, 162, 0.5);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #56ccf2 0%, #2f80ed 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
        box-shadow: 0 5px 15px rgba(47, 128, 237, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(47, 128, 237, 0.6);
    }
    
    /* Result card styling */
    .stMarkdown {
        color: white;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: white;
        font-size: 1.5rem;
    }
    
    /* Custom result box */
    .result-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .result-spam {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 5px 20px rgba(255,107,107,0.4);
        animation: slideIn 0.5s ease;
    }
    
    .result-safe {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 5px 20px rgba(81,207,102,0.4);
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Performance metrics box */
    .metrics-box {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load dataset
data = pd.read_csv("spam.csv")  # Use relative path
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Prepare data
mess = data['Message'].tolist()
cat = data['Category'].tolist()

# Split the data with random_state for reproducibility
mess_train, mess_test, cat_train, cat_test = train_test_split(
    mess, cat, test_size=0.2, random_state=42
)

# Vectorize text data
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Create and train the model
model = MultinomialNB()
model.fit(features, cat_train)

# Test the model
features_test = cv.transform(mess_test)
accuracy = model.score(features_test, cat_test) * 100

# Calculate precision
predictions = model.predict(features_test)
precision = precision_score(cat_test, predictions, pos_label='Spam') * 100

# Streamlit App
st.header('📧 Spam Email Detection System')
st.subheader('🔍 Enter a message below to check if it is spam or not:')

# Predict function
def predict(message):
    message_input = cv.transform([message]).toarray()
    result = model.predict(message_input)
    return result

# User Input and Validation
input_mess = st.text_input('Enter your message here', placeholder='Type your message...')

if st.button('Validate'):
    if input_mess.strip():
        output = predict(input_mess)
        
        # Display result with styled box
        st.markdown("<br>", unsafe_allow_html=True)
        
        if output[0] == 'Spam':
            st.markdown(f"""
            <div class="result-spam">
                🚨 SPAM DETECTED
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                ✅ SAFE MESSAGE
            </div>
            """, unsafe_allow_html=True)
        
        # Display Accuracy and Precision
        st.markdown("---")
        st.subheader('📊 Model Performance:')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2f}%")
        with col2:
            st.metric("Precision", f"{precision:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    else:
        st.warning("⚠️ Please enter a message to validate")
