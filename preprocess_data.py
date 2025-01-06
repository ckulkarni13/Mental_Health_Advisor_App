import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def preprocess_and_save(file_name, output_file):
    # Load the cleaned data
    data = pd.read_csv(file_name, sep='|')

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^A-Za-z0-9\\s]', '', str(text)).lower()
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stop words and lemmatize tokens
        processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        # Join tokens back into a single string
        return ' '.join(processed_tokens)

    # Apply preprocessing to 'Context' and 'Response' columns
    data['Context'] = data['Context'].apply(preprocess_text)
    data['Response'] = data['Response'].apply(preprocess_text)

    # Save the processed data to a new TXT file
    data.to_csv(output_file, index=False, sep='|')
    
    print(f"Data successfully processed and saved to '{output_file}'")
    return data

# Preprocess the data and save it as 'processed_data_final.txt'
processed_data = preprocess_and_save('cleaned_data.txt', 'processed_data_final.txt')