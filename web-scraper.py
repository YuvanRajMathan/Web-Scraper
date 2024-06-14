import openai
import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from openai.embeddings_utils import distances_from_embeddings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded API Key: {api_key}")

# Check if the API key is set
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it or include it in the script.")
else:
    openai.api_key = api_key

# Function to clean up text by removing extra spaces and blank lines
def clean_text(text):
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    cleaned_text = ' '.join(lines)
    return cleaned_text

# Function to scrape text content from a webpage
def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve the webpage: {url}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text from specific tags, e.g., paragraphs
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])
    
    return text_content

# Function to process a URL and return cleaned text and embeddings
def process_url(url):
    try:
        text = scrape_website(url)
        tokenized_text = ' '.join(word_tokenize(text))
        cleaned_text = clean_text(tokenized_text)
        n_tokens = len(word_tokenize(cleaned_text))
        embeddings = openai.Embedding.create(input=cleaned_text, engine='text-embedding-ada-002')['data'][0]['embedding']
        return cleaned_text, n_tokens, embeddings
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, None, None

# List of URLs to scrape (for larger websites, this list can be generated dynamically)
urls = [
    'https://example.com/page1',
    'https://example.com/page2',
    'https://example.com/page3',
    # Add more URLs as needed
]

# Use ThreadPoolExecutor for parallel processing
results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_url, url): url for url in urls}
    for future in as_completed(futures):
        result = future.result()
        if result[0] is not None:
            results.append(result)

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['text', 'n_tokens', 'embeddings'])

# Write the DataFrame to a CSV file
csv_file_path = 'processed/embeddings.csv'
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
df.to_csv(csv_file_path, index=False)

# Read the CSV file and process embeddings
df = pd.read_csv(csv_file_path)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe.
    """
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values.tolist(), distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a chat completion using the question and context
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(e)
        return ""

question = input("Enter your question ?")

answer = answer_question(df, question=question)
print("Answer:", answer)

