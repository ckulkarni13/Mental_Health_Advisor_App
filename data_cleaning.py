import pandas as pd
import re
import os

def load_and_clean_data(file_name):
    try:
        # Load the dataset
        data = pd.read_csv(file_name)
        print("File loaded successfully!")
        
        # Remove special characters and unwanted symbols from 'Context' and 'Response' columns
        data['Context'] = data['Context'].apply(lambda x: re.sub(r'[^A-Za-z0-9\\s]', '', str(x)))
        data['Response'] = data['Response'].apply(lambda x: re.sub(r'[^A-Za-z0-9\\s]', '', str(x)))
        
        # Save cleaned data to a new TXT file
        output_file = 'cleaned_data.txt'
        data.to_csv(output_file, index=False, sep='|')
        
        # Confirm file creation
        if os.path.exists(output_file):
            print(f"Cleaned data saved successfully as '{output_file}'")
        else:
            print("Error: File was not saved.")
        
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
cleaned_data = load_and_clean_data('data_mental_health.csv')
