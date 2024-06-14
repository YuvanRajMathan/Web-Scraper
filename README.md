# Web-Scraper
# Web Scraper with OpenAI Embeddings

This project scrapes text content from specified web pages and generates embeddings using OpenAI's API. The embeddings are then used to find contextually similar texts and answer questions based on the scraped data.

## Features

- Scrapes text content from multiple web pages.
- Cleans and processes the scraped text.
- Generates embeddings using OpenAI's text-embedding-ada-002 model.
- Answers questions based on the context of the scraped and processed text.

## Prerequisites

- Python 3.7+
- OpenAI API Key

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/web-scraper-openai.git
    cd web-scraper-openai
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory of the project and add your OpenAI API key:

    ```plaintext
    OPENAI_API_KEY=your_api_key_here
    ```

## Usage

1. Modify the list of URLs in the script to include the web pages you want to scrape:

    ```python
    urls = [
        'https://example.com/page1',
        'https://example.com/page2',
        'https://example.com/page3',
        # Add more URLs as needed
    ]
    ```

2. Run the script:

    ```bash
    python web_scraper.py
    ```

3. Follow the prompts to ask questions based on the scraped data.

## Example

Here is an example of how to use the script:

```bash
$ python web_scraper.py
Loaded API Key: your_api_key_here
Enter your question: What is the main topic of the first page?
Answer: The main topic of the first page is...
