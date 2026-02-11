from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import random
import re
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import numpy as np
import requests
import logging
import time
from typing import List, Dict, Optional
from multiprocessing import Pool
import argparse
import os
import glob
from PyPDF2.errors import PdfReadError
from nltk.tokenize import sent_tokenize

from torch import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Article:
    title_main: str
    title: str
    text: str
    num: Optional[str] = None
    id: Optional[str] = None
    sectionParentTitre: Optional[str] = None
    
    def to_dict(self):
        return {
            "num": self.num,
            "title_main": self.title_main,
            "text": self.text,
            "id": self.id
        }


class Processor:
    def __init__(self, config_file):
        self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            self.input_file = config['input_file']
            self.endpoints = config['endpoints']
            self.num_jobs = config['num_jobs']
            self.output_file = config['output_file']
            self.flush_interval = config.get('flush_interval', 60)  # Flush interval in seconds
            self.model = config['model']
            self.max_questions = config['amount_questions']

    def process(self, log_path="processed_articles.log"):
        articles = self.load_articles(self.input_file)
        results = []
        processed_ids = []
        start_time = time.time()

        processed_articles = self.load_processed_articles(log_path)
        remaining_articles = [article for article in articles if article.id not in processed_articles]

        articles_per_endpoint = self.split_articles_on_endpoints(remaining_articles)

        with Pool(processes=self.num_jobs) as pool:
            for result in pool.imap_unordered(self.call_endpoint, articles_per_endpoint):
                results.append(result)

                processed_ids.append(result["article"].id)
                
                # Periodic saving of the results
                if time.time() - start_time >= self.flush_interval:
                    self.flush_results(results, mode='a')  # Append results
                    self.log_processed_articles(log_path, processed_ids)
                    results.clear()
                    processed_ids.clear()
                    start_time = time.time()

        # Last flush of the remaining results
        if results:
            self.flush_results(results, mode='a')


    def load_processed_articles(self, log_path: str) -> set:
        """Load the IDs of the items already processed from the log file."""
        if not os.path.exists(log_path):
            return set()
        with open(log_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)

    def log_processed_articles(self, log_path: str, processed_ids):
        """Add a processed item ID to the log file."""
        with open(log_path, 'a', encoding='utf-8') as f:
            for article_id in processed_ids:
                f.write(f"{article_id}\n")

    def get_embedding(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for a list of texts using the embedding API."""
        embeddings = []
        
                # Ensure 'texts' is a list, even if a single string is passed
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise TypeError(f"'texts' should be a list or string, got {type(texts)}")

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = requests.post(
                    "http://localhost:8000/embed",
                    json={"texts": [str(text) for text in batch_texts]},  
                    timeout=50
                )
                response.raise_for_status()
                batch_embeddings = response.json().get("embeddings", [])
                if batch_embeddings:
                    embeddings.extend(batch_embeddings)
                else:
                    logger.log(f"No embeddings returned for batch {i}")
                    return np.zeros((len(texts), 1024))
            except Exception as e:
                logger.log(f"Error getting embeddings for batch {i}: {e}")
                return np.zeros((len(texts), 1024))
        
        if not embeddings:
            return np.zeros((len(texts), 1024))
            
        return np.array(embeddings)

    def chunk_text(self, text: str, num_chunks: int = 3) -> List[str]:
        """Split text into exactly num_chunks chunks of approximately equal size."""
        # Split into sentences
        sentences = sent_tokenize(text)
        if len(sentences) < num_chunks:
            return sentences  # Return all sentences if fewer than num_chunks
            
        # Calculate target size for each chunk
        total_length = sum(len(s) for s in sentences)
        target_chunk_size = total_length // num_chunks
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_count = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            # Check if we should create a new chunk
            if current_length >= target_chunk_size and chunk_count < num_chunks - 1:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
                chunk_count += 1
        
        # Add remaining sentences to the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


    def scrape_and_process_url(self, url: str) -> str:
        """Scrape content from a URL and process it (HTML, XML, or PDF)."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=180)
            response.raise_for_status()

            # Check for PDF
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' in content_type:
                pdf_content = BytesIO(response.content)
                try:
                    reader = PdfReader(pdf_content)

                    # Extract text from each page
                    text = "\n".join(
                        page.extract_text() for page in reader.pages if page.extract_text()
                    )
                except PdfReadError as e:
                    print(f"Error reading PDF {url}: {e}", "error")
                    return ""

            # Check for XML or HTML
            elif url.endswith(".xml"):
                soup = BeautifulSoup(response.content, 'xml')
                text = soup.get_text(separator=' ', strip=True)
            else:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text content
                text = soup.get_text(separator=' ', strip=True)

            # Clean up whitespace
            text = ' '.join(text.split())
            return text

        except Exception as e:
            print(f"Error scraping URL {url}: {e}", "error")
            return ""



    def get_searxng_results(self, query: str, top_k: int = 3):
        try:
            params = {
                "q": query,
                "format": "json",
                "engines": "wiki",
                "language": "de",
                "max_results": 3
            }
   
            # Make request to SearXNG
            results = self.make_searxng_request("http://localhost:8080", params)
            
            processed_results = []
            counter = 0
            
            # Process only the first `top_k` results
            for result in results.get("results", [])[:top_k]:
                url = result.get('url', '')
             
                scraped_content = self.scrape_and_process_url(url)
         
                if scraped_content == "":
                    counter += 1
                    continue
                
                chunks = self.chunk_text(scraped_content)
                
                for chunk in chunks:

                    processed_results.append({
                        "score": 1,
                        "article": {
                            "url": url,
                            "title": result.get('title', ''),
                            "content": chunk
                        },
                        "source": "SearXNG"
                })
            

            return processed_results

        except Exception as e:
            return []

    def load_articles(self, input_path: str) -> List[Article]:
        articles = []
        data_path = Path(input_path)
        
        for file_path in data_path.glob("*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        articles.append(Article(**data))
                    except Exception as e:
                        logger.error(f"Error loading article from {file_path}: {e}", "error")  
                        
        
        return articles
    
    def split_articles_on_endpoints(self, articles: List[str]) -> List[Dict]:
        """Teilt die Artikel gleichmäßig auf die Endpoints auf."""
        endpoint_articles = []
        num_endpoints = len(self.endpoints)
        for i, article in enumerate(articles):
            # Determines the endpoint for this item
            endpoint_index = i % num_endpoints
            endpoint_articles.append({'article': article, 'endpoint': self.endpoints[endpoint_index]})
        return endpoint_articles
    
    def get_relevant_texts(self, results: List[Dict], max_token_count: int) -> List[Dict]:
        """Extract the most relevant text passages from results up to max_token_count total."""
        relevant_texts = []
        total_tokens = 0
        
    
        if not results:
            return relevant_texts
        
        random.shuffle(results)

        for result in results:
            
            article = result.get('article')
            text = article.get('content', '')
            source_info = {
                    "source": "SearXNG",
                    "url": article.get('url', ''),
                    "title": article.get('title', '')
                }
            
            # Count tokens using the API
            try:
                response = requests.post(
                    "http://localhost:8001/count_tokens",
                    json={"text": text},
                    timeout=50
                )
                response.raise_for_status()
                token_count = response.json()["count"]
            except Exception as e:
                logger.error(f"Error counting tokens: {e}")
                continue
            
            # Check if adding this text would exceed the token limit
            if total_tokens + token_count > max_token_count and token_count < 20000:
                # Calculate remaining tokens
                remaining_tokens = max_token_count - total_tokens
                if remaining_tokens > 0:
                    try:
                        # Truncate the text using the API
                        response = requests.post(
                            "http://localhost:8001/tokenize",
                            json={
                                "texts": [text],
                                "max_length": remaining_tokens,
                                "truncation": True
                            },
                            timeout=50
                        )
                        response.raise_for_status()
                        truncated_text = response.json()["texts"][0]
                        relevant_texts.append({
                            "text": truncated_text,
                            "score": result["score"],
                            "source_info": source_info
                        })
                    except Exception as e:
                        logger.error(f"Error truncating text: {e}")
                break
            
            if token_count < 20000:
                relevant_texts.append({
                    "text": text,
                    "score": result["score"],
                    "source_info": source_info
                })
                total_tokens += token_count
        

        return relevant_texts

    def create_context(self, context_texts, article: Article):
        context_parts = []
        context_parts.append(f"{article.num} {article.title_main}\n{article.text}\n")
        for ctx in context_texts:
            source_info = ctx["source_info"]

            source_str = f"[Source: {source_info['title']} - {source_info['url']}]"
        
            
            context_parts.append(f"{source_str}\n{ctx['text']}\n")
        
        context = "\n".join(context_parts)
        return context

    def call_endpoint(self, article_data: Dict) -> Dict:
        """Call FAU endpoint with fallback and retry logic."""
        article = article_data['article']
        endpoint = article_data['endpoint']
        context = []

        for query in ["Anwendung", "Interpreation", "Kommentar"]:
            query= f"{article.num} {article.title_main} {query}"   
            context.extend(self.get_searxng_results(query=query))

        snd_context = self.get_relevant_texts(context, 50000)
        
        final_context = self.create_context(snd_context, article=article)

        try:
            payload = self.prepare_payload(final_context)
            
            response = requests.post(endpoint, json=payload, timeout=600)
            response.raise_for_status()
    
            # Parse and validate the response
            response_content = response.json()["choices"][0]["message"]["content"]
            # Remove code formatting if present
            cleaned_json = re.sub(r"^```json\n|\n```$", "", response_content, flags=re.MULTILINE).strip()
            
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:].strip()
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3].strip()

            questions = json.loads(cleaned_json)
    
            if not isinstance(questions, list) or len(questions) > self.max_questions:
                raise ValueError("Invalid response format")
            
            
            return {
                    'article': article,
                    'questions': questions,
                    'context': snd_context
                    }
        
        except requests.exceptions.RequestException as e:
            return self.call_endpoint_rec(article, payload, tried_endpoints=[endpoint], snd_context=snd_context)

    def call_endpoint_rec(self, article, payload: Dict, tried_endpoints: List[str], snd_context) -> Dict:
        """Recursive method to retry with different endpoints."""
        # Find an endpoint that hasn't been tried yet
        available_endpoints = [ep for ep in self.endpoints if ep not in tried_endpoints]
        if not available_endpoints:
            return {'article': article, 'error': 'All endpoints failed'}
        
        # Select a random endpoint and add it to the tried list
        endpoint = random.choice(available_endpoints)
        tried_endpoints.append(endpoint)

        try:
            response = requests.post(endpoint, json=payload, timeout=600)
            response.raise_for_status()
    
            # Parse and validate the response
            response_content = response.json()["choices"][0]["message"]["content"]
            # Remove code formatting if present
            cleaned_json = re.sub(r"^```json\n|\n```$", "", response_content, flags=re.MULTILINE).strip()
            
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:].strip()
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3].strip()

            questions = json.loads(cleaned_json)
    
            if not isinstance(questions, list) or len(questions) > self.max_questions:
                raise ValueError("Invalid response format")
            
            
            return {
                    'article': article,
                    'questions': questions,
                    'context': snd_context
                    }
        
        except requests.exceptions.RequestException as e:
            return self.call_endpoint_rec(article, payload, tried_endpoints, snd_context)

    def prepare_payload(self, article: str) -> Dict:
        """Prepare the request payload for the FAU endpoint."""
        system_prompt = """
        You are an expert in German tax law. Your task is to generate high-quality and meaningful questions related to the given context and answer. Your task is to go into a more specific field than the old question. This means you take the given context and try to identify a detailled are of german tax law.
        Each question must belong to one of the following five predefined types. Only generate questions that are relevant to the given article. 
        Follow these detailed descriptions, examples, and quality criteria for each type:

        **1. Verständnis- bzw. Kommentarfragen**
        - Ziel: Die Fragen sollen sich auf die Anwendung oder die Interpretation des Paragraphen beziehen und spezifisch sein. 
                Dabei können auch Beispiele genutzt werden, um den Kontext besser zu verdeutlichen.
        - Beispiele: 
            - "Wie ist § 14 AO zu verstehen, insbesondere in Bezug auf die Abgrenzung zwischen gewerblichen Einkünften und anderen Einkunftsarten? Kannst du dies anhand eines Beispiels mit einem Freiberufler und einem Gewerbebetrieb erläutern?"
            - "Was ist eine verdeckte Gewinnausschüttung (vGA)? Könnte dies z. B. der Fall sein, wenn ein Geschäftsführer ein Firmenfahrzeug auch privat nutzt?"
            - "Wie funktioniert die Fünftelregelung bei der Besteuerung von Abfindungen? Kannst du das an einem Arbeitnehmer erklären, der im Jahr 2024 eine Abfindung von 50.000 € erhält?"
        - Qualitätskriterien: 
            - Klare Bennenung und verständliche Sprache.
            - WICHTIG: Es soll kein Beispielhaftes Szenario geben, sondern direkt nach einem steuerlichen Begriff gefragt werden und wie dieser Anwendung findet

        **Generelle Anforderungen:**
        - Generiere bis zu {self.max_questions} Fragen (mindestens eine).
        - Wähle nur die Fragetypen aus, die für den gegebenen Kontext wirklich sinnvoll sind.
        - Formatiere die Ausgabe als JSON-Array mit dem Feld 'frage'.
        - Die Fragen sollen ausschließlich auf Deutsch geschrieben werden.
        - Die Fragen sollen so ausführlich, wie möglich sein und wenn du Beispiele oder Szenarien nennst, sollten diese überaus detailiert sein.
        - Die Fragen sollen auf kreativen Weg formuliert werden, um den Kontext zu verstehen und die Fragen zu vermeiden, die bereits in der Antwort enthalten sind.
        - Nutze NIEMALS die Beispielfragen!
        """

        
        user_prompt = f"""
        Context: {article}
        
        Generate up to {self.max_questions} high-quality questions based on the provided context. The number of questions should depend on the available relevant information. If fewer than {self.max_questions} suitable and meaningful questions can be generated, return only those. Avoid creating any irrelevant or speculative questions and NEVER (IMPORTANT) use the example questions.
        Check for each question type, if there is a possibility to generate a question, if you are unsure skip the question type as you explicitly do not need to create exactly {self.max_questions} questions. 
        The questions should be like reallife questions and scenarios of a tax lawyer and his corporate clients.
        Do not use Names in the question, if you need to reference a person do that by calling it "Person A" or any other capital Letter after "Person", but try to rather create questions about "Unternehmen".
        Be very specific with the tax and law related terms and if suitable incorperate them in the questions.
        IMPORTANT: Do not use the same question multiple times, also watch out that you do not create the same question multiple times which simply uses different terms.
        If possible, all questions should have a different tax law topic and should not address the same issue.
        Your primary focus should be the article itself. Only if the other context helps with understanding the article better you should use it.
        Only include questions that fit the context. Format your response as a JSON array, as shown below:
        [
            {{"frage": "Example question?"}},
            ...
        ]
        """
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.5,
            "top_p": 0.9,
            "model": self.model
        }

    def clean_result(self, result: Dict) -> Dict:
        """Clean and validate the FAU endpoint result."""
        try:
            return {"answer": result["choices"][0]["message"]["content"]}
        except (KeyError, IndexError):
            return {"error": "Unexpected response format", "raw": result}

    def flush_results(self, results: List[Dict], mode: str = 'w'):
        """Flush results to the output file."""
        with open(self.output_file, mode, encoding='utf-8') as f:
            for result in results:
                # Convert 'article' to a dictionary if it exists
                if "article" in result and isinstance(result["article"], Article):
                    result["article"] = result["article"].to_dict()
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"{len(results)} results flushed to {self.output_file}")



    from requests.exceptions import RequestException

    def make_searxng_request(self, url: str, params: dict) -> dict:
        """Make a request to SearXNG with retry logic."""
        response = requests.get(url, headers={"User-Agent": "Legal-Retrieval-Bot"}, params=params)
        response.raise_for_status()
        return response.json()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()

    processor = Processor(args.config)
    processor.process()
