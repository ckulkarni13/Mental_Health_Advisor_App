import os
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "mental-health-index"

pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists
if index_name not in pc.list_indexes().names():
    st.error(f"Index '{index_name}' does not exist. Please ensure it is created and populated.")
    st.stop()

# Connect to Pinecone index
index = pc.Index(index_name)

# OpenAI Initialization
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_query_embedding(query):
    try:
        response = client.embeddings.create(model="text-embedding-ada-002", input=query)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Failed to generate query embedding: {e}")
        return None

def query_pinecone(query_embedding, pinecone_index, top_k=5):
    try:
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        if results["matches"]:
            return [match["metadata"] for match in results["matches"]]
        else:
            st.warning("No matches found in Pinecone.")
            return []
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return []

def generate_advice_with_gpt4(query, context):
    try:
        context_combined = "\n".join([c.get('response', '') for c in context])
        messages = [
            {"role": "system", "content": "You are a professional assistant providing detailed and empathetic advice."},
            {"role": "user", "content": f"""
            You are an empathetic and professional assistant providing support to mental health counselors.
            Based on the query and the retrieved context, provide a detailed, empathetic, and actionable response.

            User Query: {query}

            Retrieved Context:
            {context_combined}

            Response:
            """}
        ]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response with GPT-4: {e}")
        return "Sorry, I couldn't generate a response at the moment."

def handle_user_query(user_query, pinecone_index):
    query_embedding = generate_query_embedding(user_query)
    if not query_embedding:
        return "Failed to generate embedding for the query."

    retrieved_context = query_pinecone(query_embedding, pinecone_index, top_k=5)
    if not retrieved_context:
        return "No relevant data found for the query."

    final_advice = generate_advice_with_gpt4(user_query, retrieved_context)
    return final_advice


# Streamlit UI
st.title("Mental Health Counselor Assistant")

st.write("This application helps mental health counselors by providing detailed, empathetic, and actionable advice based on user queries.")

# Add a dropdown for mental health issues
st.subheader("Select a Mental Health Issue")
mental_health_issues = ["Depression", "Anxiety", "PTSD", "Bipolar Disorder", "Schizophrenia"]
selected_issue = st.selectbox("Choose an issue or type your own query below:", mental_health_issues + ["Other"])

# Modify the user query input
user_query = ""
if selected_issue == "Other":
    user_query = st.text_input("Enter your specific query:")
else:
    user_query = f"Provide advice for treating {selected_issue}"

if st.button("Generate Advice"):
    if user_query.strip():
        with st.spinner("Processing your query..."):
            advice = handle_user_query(user_query, index)
            st.subheader(f"Generated Advice for {selected_issue}:")
            st.write(advice)
    else:
        st.warning("Please select an issue or enter a query.")

# import os
# from openai import OpenAI
# from pinecone import Pinecone
# import streamlit as st
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Initialize Pinecone
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "mental-health-index"
# index = pc.Index(index_name)

# # OpenAI API setup
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Function to generate embeddings for a query
# def generate_query_embedding(query):
#     try:
#         response = client.embeddings.create(model="text-embedding-ada-002", input=[query])
#         return response.data[0].embedding
#     except Exception as e:
#         st.error(f"Failed to generate query embedding: {e}")
#         return None

# # Function to query Pinecone for relevant vectors
# def query_pinecone(query_embedding, pinecone_index, top_k=5):
#     try:
#         results = pinecone_index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             include_metadata=True
#         )
#         if "matches" in results and results["matches"]:
#             return [match["metadata"] for match in results["matches"]]
#         else:
#             st.warning("No matches found in Pinecone.")
#             return []
#     except Exception as e:
#         st.error(f"Error querying Pinecone: {e}")
#         return []

# # Function to generate advice using GPT-4
# def generate_advice_with_gpt4(query, context):
#     try:
#         context_combined = "\n".join([c.get('response', '') for c in context])

#         messages = [
#             {"role": "system", "content": "You are a professional assistant providing detailed and empathetic advice."},
#             {"role": "user", "content": f"""
#             You are an empathetic and professional assistant providing support to mental health counselors.
#             Based on the query and the retrieved context, provide a detailed, empathetic, and actionable response.

#             User Query: {query}

#             Retrieved Context:
#             {context_combined}

#             Response:
#             """}
#         ]

#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=messages
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         st.error(f"Error generating response with GPT-4: {e}")
#         return "Sorry, I couldn't generate a response at the moment."

# # Streamlit Interface
# def main():
#     st.title("Mental Health Counselor Assistant")
#     st.write("""
#         This application helps mental health counselors by providing detailed, empathetic, 
#         and actionable advice based on user queries and retrieved context.
#     """)

#     # Input query
#     user_query = st.text_input("Enter your query:", "")

#     if st.button("Generate Advice"):
#         if user_query.strip():
#             with st.spinner("Processing your query..."):
#                 # Step 1: Generate query embedding
#                 query_embedding = generate_query_embedding(user_query)
#                 if not query_embedding:
#                     st.error("Failed to generate embedding for the query.")
#                     return

#                 # Step 2: Query Pinecone
#                 retrieved_context = query_pinecone(query_embedding, index, top_k=5)  # Fixed variable name
#                 if not retrieved_context:
#                     st.warning("No relevant data found for the query.")
#                     return

#                 # Step 3: Generate advice with GPT-4
#                 final_advice = generate_advice_with_gpt4(user_query, retrieved_context)

#                 # Display results
#                 st.subheader("Retrieved Context")
#                 for i, context in enumerate(retrieved_context, 1):
#                     st.write(f"**Context {i}:** {context.get('response', '')}")

#                 st.subheader("Generated Advice")
#                 st.success(final_advice)
#         else:
#             st.warning("Please enter a query.")

# if __name__ == "__main__":
#     main()  -- working

# import os
# from openai import OpenAI
# from pinecone import Pinecone
# import streamlit as st
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "mental-health-index"
# index = pc.Index(index_name)

# # OpenAI API setup
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def generate_query_embedding(query):
#     try:
#         response = client.embeddings.create(model="text-embedding-ada-002", input=[query])
#         return response.data[0].embedding
#     except Exception as e:
#         st.error(f"Failed to generate query embedding: {e}")
#         return None

# # Function to query Pinecone for relevant vectors
# def query_pinecone(query_embedding, pinecone_index, top_k=5):
#     try:
#         results = pinecone_index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             include_metadata=True
#         )
#         if "matches" in results and results["matches"]:
#             return [match["metadata"] for match in results["matches"]]
#         else:
#             st.warning("No matches found in Pinecone.")
#             return []
#     except Exception as e:
#         st.error(f"Error querying Pinecone: {e}")
#         return []

# # Function to generate advice using GPT-4
# def generate_advice_with_gpt4(query, context):
#     try:
#         context_combined = "\n".join([c.get('response', '') for c in context])

#         messages = [
#             {"role": "system", "content": "You are a professional assistant providing detailed and empathetic advice."},
#             {"role": "user", "content": f"""
#             You are an empathetic and professional assistant providing support to mental health counselors.
#             Based on the query and the retrieved context, provide a detailed, empathetic, and actionable response.

#             User Query: {query}

#             Retrieved Context:
#             {context_combined}

#             Response:
#             """}
#         ]

#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=messages
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         st.error(f"Error generating response with GPT-4: {e}")
#         return "Sorry, I couldn't generate a response at the moment."

# # Streamlit Interface
# def main():
#     st.title("Mental Health Counselor Assistant")
#     st.write("""
#         This application helps mental health counselors by providing detailed, empathetic, 
#         and actionable advice based on user queries and retrieved context.
#     """)

#     # Input query
#     user_query = st.text_input("Enter your query:", "")

#     if st.button("Generate Advice"):
#         if user_query.strip():
#             with st.spinner("Processing your query..."):
#                 # Step 1: Generate query embedding
#                 query_embedding = generate_query_embedding(user_query)
#                 if not query_embedding:
#                     st.error("Failed to generate embedding for the query.")
#                     return

#                 # Step 2: Query Pinecone
#                 retrieved_context = query_pinecone(query_embedding, pinecone_index, top_k=5)
#                 if not retrieved_context:
#                     st.warning("No relevant data found for the query.")
#                     return

#                 # Step 3: Generate advice with GPT-4
#                 final_advice = generate_advice_with_gpt4(user_query, retrieved_context)

#                 # Display results
#                 st.subheader("Retrieved Context")
#                 for i, context in enumerate(retrieved_context, 1):
#                     st.write(f"**Context {i}:** {context.get('response', '')}")

#                 st.subheader("Generated Advice")
#                 st.success(final_advice)
#         else:
#             st.warning("Please enter a query.")

# if __name__ == "__main__":
#     main()