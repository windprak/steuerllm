from io import BytesIO
from PyPDF2 import PdfReader
import sys
import numpy as np
import json
import os
import logging
import requests
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import time
import re
from datetime import datetime, timedelta
import argparse
import multiprocessing
from multiprocessing import Pool, Value
from dataclasses import dataclass, asdict
import json
from PyPDF2.errors import PdfReadError

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

headers = {
    'Content-Type': 'application/json',
}

searxng_url = os.getenv('SEARXNG_ENDPOINT', "http://localhost:8080")

@dataclass
class Article:
    title_main: str
    title: str
    text: str
    num: Optional[str] = None
    id: Optional[str] = None
    sectionParentTitre: Optional[str] = None

@dataclass
class JobConfig:
    job_id: str
    input_file: str
    output_file: str
    fau_endpoint: str
    searxng_endpoint: str
    embeddings_cache: str
    max_output_file_size_gb: str
    model: str
    flush_settings: dict
    runtime_settings: dict
    resource_limits: dict
    enabled: bool = True

    def get_max_output_file_size_gb(self) -> float:
        return float(self.max_output_file_size_gb)

    def get_flush_intervals(self) -> tuple[int, int]:
        return (
            int(self.flush_settings['qa_results_interval_minutes']),
            int(self.flush_settings['questions_interval_minutes'])
        )

    def get_runtime_settings(self) -> tuple[int, float, float]:
        return (
            int(self.runtime_settings['duration_hours']),
            float(self.runtime_settings['sleep_after_question_seconds']),
            float(self.runtime_settings['sleep_after_generation_seconds'])
        )

    def get_resource_limits(self) -> tuple[float, float]:
        return (
            float(self.resource_limits['max_memory_gb']),
            float(self.resource_limits['max_cpu_percent'])
        )


def get_embedding(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Get embeddings for a list of texts using the embedding API."""
    embeddings = []
    
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
                print(f"No embeddings returned for batch {i}")
                return np.zeros((len(texts), 1024))
        except Exception as e:
            print(f"Error getting embeddings for batch {i}: {e}")
            return np.zeros((len(texts), 1024))
    
    if not embeddings:
        return np.zeros((len(texts), 1024))
        
    return np.array(embeddings)

def scrape_and_process_url(url: str) -> str:
    """Scrape content from a URL and process it (HTML, XML, or PDF)."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=180) #TODO: Set back to 15
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
                log_message(f"Error reading PDF {url}: {e}", "error")
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
        log_message(f"Error scraping URL {url}: {e}", "error")
        return ""


def chunk_text(text: str, num_chunks: int = 3) -> List[str]:
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


def get_searxng_results(query: str, top_k: int = 3):
    """Get search results from SearXNG, scrape results, and calculate similarity scores."""
    try:
        params = {
            "q": query,
            "format": "json",
            "engines": "wiki",
            "language": "de",
            "max_results": 6
        }

        # Make request to SearXNG
        results = make_searxng_request(searxng_url, params)
        
        # Get query embedding
        query_embedding = get_embedding([query])
        
        processed_results = []

        counter = 0
        
        # Process only the first `top_k` results
        for idx, result in enumerate(results.get("results", [])[:top_k]):
            url = result.get('url', '')
            scraped_content = scrape_and_process_url(url)
            
            if scraped_content == "":
                counter += 1
                continue
            
            chunks = chunk_text(scraped_content)
            
            for chunk in chunks:
                chunk_embedding = get_embedding([chunk])[0]
                similarity = cosine_similarity(query_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0]

                processed_results.append({
                    "score": float(similarity),
                    "article": {
                        "url": url,
                        "title": result.get('title', ''),
                        "content": chunk
                    },
                    "source": "SearXNG"
            })
        
        # Process remaining results if some were skipped
        if counter > 0:
            end = top_k + counter
            for idx, result in enumerate(results.get("results", [])[top_k:end]):
                url = result.get('url', '')
                scraped_content = scrape_and_process_url(url)
                
                if scraped_content == "":
                    continue
                
                chunks = chunk_text(scraped_content)
                
                for chunk in chunks:
                    chunk_embedding = get_embedding([chunk], 240)[0]
                    similarity = cosine_similarity(query_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0]

                    processed_results.append({
                        "score": float(similarity),
                        "article": {
                            "url": url,
                            "title": result.get('title', ''),
                            "content": chunk
                        },
                        "source": "SearXNG"
                    })
        
        # If no valid results, retry with a new query
        if not processed_results:
            query = generate_new_query(query)
            processed_results = get_searxng_results(query, top_k)

        return processed_results

    except Exception as e:
        log_message(f"Error getting SearXNG results: {e}", "error")
        return []

def generate_new_query(query: str):
            # Prepare messages for chat completion
        messages = [
            {"role": "system", "content": "Du erhältst immer einen Kontext in Form einer alten Suchmaschinen-Query, die zu keinem Ergebnis geführt hat. Deine Aufgabe ist es, basierend auf den wichtigen Aspekten der alten Query eine neue Such-Query zu erstellen, die sich auf steuerrechtliche Fragestellungen konzentriert und für die Suchmaschine besser verständlich ist. Extrahiere die relevanten Informationen aus der ursprünglichen Query und optimiere die neue Query so, dass sie steuerrechtliche Aspekte klar hervorhebt. Gib ausschließlich die neue Query zurück, ohne zusätzliche Kommentare, Erläuterungen oder Formatierungen. Beispiel: Alter Kontext: 'Die Bäcker- & Konditormeister Müller und Weber interessieren sich für einen Volkswagen Transporter, den sie als Auslieferungswagen anschaffen möchten. Da sie den Wagen nicht bar bezahlen können, schlägt Ihnen der Volkswagen-Händler eine entsprechende Kredit- oder Leasingfinanzierung vor. Erläutern Sie bitte die wesentlichen Bedingungen eines Leasingvertrages.' Neue Query: 'steuerrechtliche Behandlung Leasingvertrag'"},
            {"role": "user", "content": f"Kontext:\n{query}\n Frage: Erstelle mir dazu eine neue Query für Search Enginges und fokussiere dich dabei darauf, dass es um das Steuerrecht geht. Gebe mir nur die neue query aus und sonst gar nichts."}
        ]
        
        # Make API call with retry logic
        response_data = make_fau_request(url, headers, {
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.2,
            "top_p": 0.9,
            "model": llm_model
        })
        
        answer = response_data["choices"][0]["message"]["content"]
        return answer


def merge_and_rank_results(searxng_results: List[dict], top_k: int = 20) -> List[dict]:
    """Merge and rank results from both sources."""
    ranked_results = sorted(searxng_results, key=lambda x: x["score"], reverse=True)
    return ranked_results[:top_k]

def get_relevant_texts(results: List[Dict], max_token_count: int = 80000) -> List[Dict]:
    """Extract the most relevant text passages from results up to max_token_count total."""
    relevant_texts = []
    total_tokens = 0

    for result in results:
        article = result["article"]
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
            print(f"Error counting tokens: {e}")
            continue
        
        # Check if adding this text would exceed the token limit
        if total_tokens + token_count > max_token_count and token_count < 30000:
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
                    print(f"Error truncating text: {e}")
            break
        
        if token_count < 30000:
            relevant_texts.append({
                "text": text,
                "score": result["score"],
                "source_info": source_info
            })
            total_tokens += token_count
    
    return relevant_texts

def get_quality_criteria(fragetyp: str):
    if fragetyp == "Verständnis- bzw. Kommentarfragen":
        return (
            "Klare Definition der angesprochenen Konzepte oder Paragraphen in verständlicher Sprache. "
            "Verwendung einfacher, nachvollziehbarer Beispiele, um den Kontext zu erklären. "
            "Erklärungen sollten ausführlich und logisch aufgebaut sein, mit zusätzlichen Details, falls notwendig."
        )
    elif fragetyp == "Buchungssätze / Bilanzkonten":
        return (
            "Präzise Beschreibung der relevanten Buchungssätze und Bilanzkonten. "
            "Berücksichtigung der Buchungslogik (Soll und Haben) mit direktem Bezug zum Geschäftsvorfall. "
            "Zusätzliche Hinweise, welche Kontenarten (Aktiv, Passiv, Aufwands-, Ertragskonten) betroffen sind."
        )
    elif fragetyp == "Rahmenfragen":
        return (
            "Strukturierte Analyse des vorgegebenen Kontexts mit klarer Darstellung der wichtigen Aspekte. "
            "Detaillierte Erklärung, welche rechtlichen, steuerlichen oder buchhalterischen Konsequenzen zu beachten sind. "
            "Falls notwendig, klare Unterscheidung und Erklärung verschiedener Szenarien."
        )
    elif fragetyp == "Korrekturfragen":
        return (
            "Klare Beurteilung der Korrektheit der angegebenen Lösung (Ja/Nein). "
            "Ausführliche Begründung, warum die Lösung korrekt oder nicht korrekt ist, mit Bezug auf relevante Vorschriften. "
            "Schritt-für-Schritt-Erklärung, um den Denkprozess nachvollziehbar zu machen."
        )
    elif fragetyp == "Quellenabfragen":
        return (
            "Logische und strukturierte Identifizierung relevanter Gesetzestexte oder Paragraphen. "
            "Jede angegebene Quelle sollte kurz erklärt werden, warum sie relevant ist. "
            "Vermeidung von irrelevanten oder unwichtigen Paragraphen, um den Fokus zu behalten."
        )
    elif fragetyp == "Ergänzungsfragen":
        return (
            "Klare Darstellung des definierten Ziels (z. B. Steuerlastsenkung, Gewinnmaximierung). "
            "Ausführliche Beschreibung der bereits vorgeschlagenen Lösung mit Step-by-Step-Erklärung. "
            "Ermittlung zusätzlicher Optimierungsmöglichkeiten, logisch sortiert und mit Bedingungen (z. B. 'wenn ... dann')."
        )
    else:
        return "Beantworte die Frage detailliert und strukturiert."



def generate_answer_with_fau(query: str, context_texts: List[Dict], fragetyp: str):
    """Generate an answer using FAU endpoint based on the query and context texts."""
    try:
        # Prepare context string
        context_parts = []
        for ctx in context_texts:
            source_info = ctx["source_info"]

            source_str = f"[Source: {source_info['title']} - {source_info['url']}]"
        
            
            context_parts.append(f"{source_str}\n{ctx['text']}\n")
        
        context = "\n".join(context_parts)
        
        if fragetyp != "":
            qualität = get_quality_criteria(fragetyp)
        else:
            qualität = "Beantworte die Frage detailliert und strukturiert."

        # Prepare messages for chat completion
        messages = [
            {"role": "system", "content": "Du bist ein hilfreicher Assistent für juristische Fragen. Beantworte die Frage basierend auf dem gegebenen Kontext. Wenn du dir bei der Antwort nicht sicher bist, gib dies an. Vermeide Aussagen, wie 'basierend auf gegebenem Kontext' oder 'aus dem gegebenem Text'. Wichtig ist, dass du sofern möglich wichtige Paragraphen, Rechtstexte oder Urteile zitierst."},
            {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {query}\n\nQualitätskriterien: {qualität}\n\n. Wenn die Antwort nicht aus dem Kontext hervorgeht, antworte nur mit: 'Es tut mir leid, aber es der Kontext ist nicht ausreichend.'. Zitiere wenn möglich Rechtstexte, Paragraphen, Gerichtsurteile oder andere staatliche offiziele Quellen. Nutze nur Quellen die sich explizit auf Deutschland beziehen, nutze keine Quellen aus Österreich oder der Schweiz."}
        ]
        
        # Make API call with retry logic
        response_data = make_fau_request(url, headers, {
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "model": llm_model
        })
        
        answer = response_data["choices"][0]["message"]["content"]
        return answer
            
    except Exception as e:
        log_message(f"Error generating answer: {e}", "error")
        
        return "Es tut mir leid, aber es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."

def load_questions(file_path: str) -> List[dict]:
    """Load questions from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_questions(file_path: str, questions: List[dict]):
    """Save remaining questions back to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

def save_qa_result(output_file: str, qa_results: List[Dict], volume, max_size_gb):
    """Save Q&A results to output file."""
    # Create file with initial list if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    else:
        current_size_gb = os.path.getsize(output_file) / (1024 ** 3)  # Größe in GB
        log_message(f"Current file size: {current_size_gb:.10f} GB, Max size allowed: {max_size_gb} GB", "info")
        
        if current_size_gb >= max_size_gb:
                # Erstelle eine neue Ausgabedatei mit einer volumesnummer
            base, ext = os.path.splitext(output_file)
            
            if volume > 0:
                base_parts = base.split("_")
                base = "_".join(base_parts[:-1])

            new_output_file = f"{base}_v{volume}{ext}"
            volume += 1
            output_file = new_output_file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            log_message(f"Successfully set new output file: {str(output_file)}", "info")
            

    # Read existing content
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_results = json.load(f)
    
    # Append new results
    existing_results.extend(qa_results)
    
    # Write back to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=4)

    return output_file, volume

def process_question(question_data: dict) -> Tuple[str, List[Dict]]:
    """Process a single question and return answer with sources."""
    question = question_data["frage"]
    
    fragetyp = question_data.get("typ", "")
    
    # Initialize results
    all_results = []
    
    # Generate new query prompt
    question = generate_new_query(question)

    
    # Get results from SearXNG
    searxng_results = get_searxng_results(query=question) #TODO wieder ohne quelle machen
    
    # Merge all results
    all_results.extend(merge_and_rank_results(searxng_results))
    
    # Get relevant texts with max token count
    max_token_count = 50000  # Set a fixed maximum token count
    context_texts = get_relevant_texts(all_results, max_token_count)
    
    # Generate answer
    answer = generate_answer_with_fau(question, context_texts, fragetyp)
    
    return answer, context_texts

def generate_more_questions(context: List[Dict], question: str, max_questions: int = 3) -> List[dict]:
    """
    Generate up to max_questions diverse tax-related questions based on the context and answer.
    The questions correspond to predefined question types, meeting specific quality criteria.
    """
    context_text = " ".join([f"{source['text']}" for source in context])

    # System prompt with detailed explanation of question types
    system_prompt = """
    You are an expert in German tax law. Your task is to generate high-quality and meaningful questions related to the given context and answer. Your task is to go into a more specific field than the old question. This means you take the given context and try to identify a detailled are of german tax law.
    Each question must belong to one of the following six predefined types. Only generate questions that are relevant to the context. 
    Follow these detailed descriptions, examples, and quality criteria for each type:

    **1. Verständnis- bzw. Kommentarfragen**
    - Ziel: Die Fragen sollen sich auf Paragraphen, Begriffe oder steuerrechtliche Konzepte beziehen und spezifisch sein. 
            Dabei können auch Beispiele genutzt werden, um den Kontext besser zu verdeutlichen.
    - Beispiele: 
        - "Wie ist § 14 AO zu verstehen, insbesondere in Bezug auf die Abgrenzung zwischen gewerblichen Einkünften und anderen Einkunftsarten? Kannst du dies anhand eines Beispiels mit einem Freiberufler und einem Gewerbebetrieb erläutern?"
        - "Was ist eine verdeckte Gewinnausschüttung (vGA)? Könnte dies z. B. der Fall sein, wenn ein Geschäftsführer ein Firmenfahrzeug auch privat nutzt?"
        - "Wie funktioniert die Fünftelregelung bei der Besteuerung von Abfindungen? Kannst du das an einem Arbeitnehmer erklären, der im Jahr 2024 eine Abfindung von 50.000 € erhält?"
    - Qualitätskriterien: 
        - Klare Bennenung und verständliche Sprache.
        - WICHTIG: Es soll kein Beispielhaftes Szenario geben, sondern direkt nach einem steuerlichen Begriff gefragt werden und wie dieser Anwendung findet


    **2. Buchungssätze / Bilanzkonten**
    - Ziel: Die Fragen sollen sich auf konkrete Buchungsvorgänge oder betroffene Bilanzkonten beziehen. 
            Typisch sind Fallbeispiele, bei denen die Verbuchung eines Geschäftsvorfalls abgefragt wird.
    - Beispiele: 
        - "Ein Unternehmen kauft einen LKW für 85.000 € netto. Der Kauf wird zu 50 % über ein Bankdarlehen und zu 50 Prozent aus der Kasse finanziert. Wie ist dieser Geschäftsvorfall zu buchen, und welche Konten werden dabei angesprochen?"
        - "Unternehmen XY erhält eine Rückzahlung von 5.000 € aus einer fehlerhaften Zahlung an einen Lieferanten. Wie wird diese Rückzahlung in der Buchhaltung erfasst?"
        - "Ein Bauunternehmen kauft eine Baumaschine für 120.000 € netto. Die Zahlung erfolgt zu 80 Prozent per Banküberweisung, der Rest wird über einen Lieferantenkredit finanziert. Wie lautet der korrekte Buchungssatz, und welche Konten sind betroffen?"
    - Qualitätskriterien: 
        - Präzise Fragestellung zum Buchungssatz.


    **3. Rahmenfragen**
    - Ziel: Ermittlung von wichtigen Aspekten, die bei einem spezifischen steuerrechtlichen Kontext berücksichtigt werden müssen.
    - Beispiele:
        - "Ein Unternehmer möchte seine GmbH an seinen Sohn übertragen. Welche steuerlichen und rechtlichen Rahmenbedingungen sollten geprüft werden, insbesondere im Hinblick auf Schenkungs- und Erbschaftssteuer?"
        - "Unternehmen ABC plans, zum Jahresende Sonderzahlungen an die Mitarbeiter in Höhe von insgesamt 500.000 € zu leisten. Welche steuerlichen Aspekte sind dabei zu beachten, und wie können diese Zahlungen optimal gestaltet werden?"
        - "Ein mittelständisches Unternehmen plant, an die Gesellschafter 5 Mio. € aus nicht entnommenen Gewinnen auszuschütten. Welche steuerlichen Auswirkungen hat dies auf die Gesellschaft und die Gesellschafter, und welche Alternativen könnten in Betracht gezogen werden?"
        - "Ein Unternehmer plant, ein Wohnhaus in ein Bürogebäude umzuwandeln und anschließend gewerblich zu nutzen. Welche steuerlichen Konsequenzen ergeben sich aus dieser Umwidmung, insbesondere im Hinblick auf die Umsatzsteuer und die Gewerbesteuer?"
    - Qualitätskriterien:
        - Strukturiert dargestellter Sachverhalt mit einer kreativen Fragestellung.
        - Fiktive Beispiele oder praktische Szenarien.
        - Frage nach dem steuerlichen Rahmen oder der steuerlichen Verhaltung (kreativ formuliert).
        - Szenario mindestens 2 Sätze lang

    **4. Korrekturfragen**
    - Ziel: Überprüfung einer vorgeschlagenen Lösung. Die Frage soll klären, ob die Lösung korrekt ist oder nicht, und bei Bedarf die Begründung liefern.
    - Beispiele:
        - "Ein Unternehmen setzt 200.000 € für den Bau eines neuen Bürogebäudes als sofort abziehbare Betriebsausgabe an. Ist dies korrekt? Wenn nicht, wie müsste die steuerliche Behandlung dieses Betrags erfolgen?"
        - "Privatperson XY hat für beruflich genutzte Räume in ihrer Eigentumswohnung pauschal 20 Prozent der Gesamtkosten als Werbungskosten abgesetzt. Ist dieser Ansatz korrekt, oder welche Anforderungen sind für die Anerkennung eines häuslichen Arbeitszimmers zu erfüllen?"
        - "Ein Freiberufler schreibt die Kosten für die Renovierung seines gesamten Wohnhauses als Werbungskosten ab, da er dort ein Arbeitszimmer nutzt. Ist diese Vorgehensweise korrekt, oder wie müsste die steuerliche Behandlung korrekt aussehen?"
    - Qualitätskriterien:
        - Klare Möglichkeit zur booleschen Bewertung ("Ja/Nein").
        - Strukturiert dargestellter Sachverhalt mit einer kreativen Fragestellung.
        - Fiktive Beispiele oder praktische Szenarien.
        - Szenario mindestens 2 Sätze lang

    **5. Quellenabfragen**
    - Ziel: Fragen sollen Gesetzestexte, Paragraphen oder andere Quellen für ein Problem identifizieren.
    - Beispiele:
        - "Ein Selbstständiger möchte den Kauf eines teuren Laptops steuerlich geltend machen. Welche Vorschriften regeln die Sofortabschreibung von geringwertigen Wirtschaftsgütern (GWG) und die Abschreibung teurerer Geräte?"
        - "Ein Arbeitnehmer erhält eine Erstattung für ein beruflich genutztes Elektroauto. Welche Paragraphen im Einkommensteuergesetz sind relevant, um die steuerliche Behandlung dieser Erstattung zu prüfen?"
        - "Ein Immobilienbesitzer möchte prüfen, ob die Einnahmen aus einer befristeten Vermietung unter die Kleinunternehmerregelung nach § 19 UStG fallen. Welche Paragraphen sind hier von Bedeutung, und wie ist die Regelung anzuwenden?"
    - Qualitätskriterien:
        - Strukturiert dargestellter Sachverhalt mit einer kreativen Fragestellung.
        - Fiktive Beispiele oder praktische Szenarien.
        - Explizite Nachfrage nach relevanten Paragraphen oder Gesetzestexten.

    **6. Ergänzungsfragen**
    - Ziel: Die Fragen sollen auf ein bestehendes Szenario mit einer bereits umgesetzten Lösung eingehen. Ziel ist es, eine Optimierung oder Verbesserung zu identifizieren, um ein definiertes Ziel – wie z. B. die Senkung der Steuerlast, die Liquiditätsverbesserung oder die Nutzung steuerlicher Vorteile – besser zu erreichen. Die Fragestellung soll nach zusätzlichen Möglichkeiten suchen, um das Ziel noch weiter zu verfolgen.
    - Beispiele:
        - "Ein mittelständisches Unternehmen hat bereits beschlossen, seinen Mitarbeitenden steuerfreie Essensgutscheine und Jobtickets anzubieten, um die Arbeitgeberattraktivität zu steigern. Welche weiteren steuerfreien oder steuervergünstigten Sachleistungen könnte das Unternehmen seinen Mitarbeitenden anbieten, um die Motivation und Bindung zu erhöhen, ohne dabei die Steuerlast erheblich zu steigern?"
        - "Ein selbstständiger Fotograf hat in diesem Jahr bereits mehrere Investitionen in neue Arbeitsgeräte getätigt und die Sofortabschreibung für geringwertige Wirtschaftsgüter (GWG) genutzt. Was könnte der Fotograf zusätzlich tun, um die Steuerlast für dieses Jahr zu reduzieren? Wären freiwillige Vorauszahlungen auf die Einkommenssteuer sinnvoll, oder könnten weitere Anschaffungen in das kommende Jahr vorgezogen werden?"
        - "Ein mittelständisches Unternehmen hat bereits beschlossen, seinen Mitarbeitenden steuerfreie Essensgutscheine und Jobtickets anzubieten, um die Arbeitgeberattraktivität zu steigern. Welche weiteren steuerfreien oder steuervergünstigten Sachleistungen könnte das Unternehmen seinen Mitarbeitenden anbieten, um die Motivation und Bindung zu erhöhen, ohne dabei die Steuerlast erheblich zu steigern?"
    - Qualitätskriterien:
        - Klare Zieldefinition (z. B. Steueroptimierung, Gewinnmaximierung).
        - Fiktive Beispiele oder praktische Szenarien.
        - Klare Angabe der aktuellen Situation (z. B. aktuelle Steuerlast) und der gefundenen Lösung. Frage nach einer Verbesserung.
        - Szenario mindestens 2 Sätze lang

    **Generelle Anforderungen:**
    - Generiere bis zu {max_questions} Fragen (mindestens eine).
    - Wähle nur die Fragetypen aus, die für den gegebenen Kontext wirklich sinnvoll sind.
    - Formatiere die Ausgabe als JSON-Array mit den Feldern 'frage', 'typ' und 'quelle'.
    - Lasse das Feld 'quelle' immer leer.
    - Die Fragen sollen ausschließlich auf Deutsch geschrieben werden.
    - Die Fragen sollen der ursprünglichen Frage nicht sehr ähnlich sein.
    - Nutze niemals die Beispielfragen, sondern lerne von ihnen.
    - Die Fragen sollen so ausführlich, wie möglich sein und wenn du Beispiele oder Szenarien nennst, sollten diese überaus detailiert sein.
    - Die Fragen sollen auf kreativen Weg formuliert werden, um den Kontext zu verstehen und die Fragen zu vermeiden, die bereits in der Antwort enthalten sind.
    """

    # User prompt specifying the task
    user_prompt = f"""
    Context: {context_text}
    Old question: {question}
    
    Generate up to {max_questions} questions, the amount of questions depends on the context and if there is a possibility to create a question. The questions should be detailed and (only) if needed contain a creative case scenario. The questions should be high-quality tax-related questions based on the context. Check for each question type, if there is a possibility to generate a question, if you are unsure skip the question type as you explicitly do not need to create exactly {max_questions} questions. 
    The questions should be like reallife questions and scenarios of a tax lawyer and his corporate clients.
    If possible, the questions should dive into another topic or a more specified field as the old question. Do not use Names in the question, if you need to reference a person do that by calling it "Person A" or any other capital Letter after "Person", but try to rather create questions about "Unternehmen".
    Be very specific with the tax and law related terms and if suitable incorperate them in the questions.
    IMPORTANT: Do not use the same question multiple times, also watch out that you do not create the same question multiple times which simply uses different terms.
    If possible, all questions should have a different tax law topic and should not address the same issue.
    Only include questions that fit the context. Format your response as a JSON array, as shown below:
    [
        {{"frage": "Example question?", "typ": "Type of the Question (e.g. "Ergänzungsfrage")", "quelle": "Should be empty always"}},
        ...
    ]
    """

    # Send the request to the endpoint
    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.5,
                "top_p": 0.9,
                "max_tokens": 1500
            },
        )
        
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
        
        if not isinstance(questions, list) or len(questions) > max_questions:
            raise ValueError("Invalid response format")
        
        for q in questions:
            if not isinstance(q, dict) or "frage" not in q or "quelle" not in q:
                raise ValueError("Invalid question format")
        
        return questions

    except (requests.RequestException, ValueError) as e:
        log_message(f"Error generating questions: {e}", "error")
        
        return []

def flush_questions(args, questions, generated_questions):
    save_questions(args, questions + generated_questions)
    
    last_questions_flush = datetime.now()
    log_message("Current questions successfully flushed", "info")
    

    return generated_questions, last_questions_flush

from requests.exceptions import RequestException

def retry_with_backoff(func):
    """Decorator to retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        max_attempts = 10
        attempt = 0
        backoff = 5  # Initial backoff in seconds
        
        while attempt < max_attempts:
            try:
                return func(*args, **kwargs)
            except RequestException as e:
                attempt += 1
                if attempt == max_attempts:
                    log_message(f"Max retry attempts ({max_attempts}) reached.", "error")
                    
                    raise
                
                wait_time = backoff * (2 ** (attempt - 1))  # Exponential backoff
                log_message(f"Connection error.", "error")
                sys.stdout.write(f"\rRetrying in {wait_time} seconds... (Attempt {attempt}/{max_attempts})")
                sys.stdout.flush()
                
                time.sleep(wait_time)
        
        return None
    return wrapper

@retry_with_backoff
def make_searxng_request(url: str, params: dict) -> dict:
    """Make a request to SearXNG with retry logic."""
    response = requests.get(url, headers={"User-Agent": "Legal-Retrieval-Bot"}, params=params)
    response.raise_for_status()
    return response.json()

@retry_with_backoff
def make_fau_request(url: str, headers: dict, data: dict) -> dict:
    """Make a request to FAU endpoint with retry logic."""
    response = requests.post(url, headers=headers, json=data, timeout=900)
    response.raise_for_status()
    return response.json()

def log_message(message: str, job_id: str, type: str="info"):
    """Logs a message."""
    if type == "info":
        logging.info(f"[Job {job_id}] {message}")
    elif type == "error":
        logging.error(f"[Job {job_id}] {message}")
    elif type == "warning":
        logging.warning(f"[Job {job_id}] {message}")

def process_job(args):
    """Process a single job configuration."""
    global url, llm_model, batch
    
    # Set global variables
    job_config = args
    url = job_config.fau_endpoint
    llm_model = job_config.model
    batch = job_config.input_file.split(".")[0].split("_")[1]
    
    # Get settings with proper types
    max_output_file_size_gb = job_config.get_max_output_file_size_gb()
    qa_flush_interval, questions_flush_interval = job_config.get_flush_intervals()
    runtime_duration, sleep_after_question, sleep_after_generation = job_config.get_runtime_settings()
    max_memory_gb, max_cpu_percent = job_config.get_resource_limits()
    
    # Initialize timing variables
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=runtime_duration)
    last_qa_flush = datetime.now()
    last_questions_flush = datetime.now()
    
    # Initialize result buffers
    qa_results_buffer = []
    generated_questions = []
    volume = 0
    
    # Load initial questions
    questions = load_questions(job_config.input_file)
    if not questions:
        log_message("No questions to process", job_config.job_id, "info")
        return

    while datetime.now() < end_time:
        if not questions:
            # Try to load more questions
            generated_questions, last_questions_flush = flush_questions(job_config.input_file, questions, generated_questions)
            generated_questions = []
            questions = load_questions(job_config.input_file)
            if not questions:
                log_message("No more questions to process", job_config.job_id, "info")
                if qa_results_buffer:
                    save_qa_result(job_config.output_file, qa_results_buffer, volume, max_output_file_size_gb)
                break

        # Process current question
        question_data = questions.pop(0)
        answer, sources = process_question(question_data)
        
        time.sleep(sleep_after_question)

        # Store result in buffer
        qa_results_buffer.append({
            "question": question_data["frage"],
            "answer": answer,
            "sources": sources
        })
        
        # Generate more questions based on the answer
        if sources:
            new_questions = generate_more_questions(sources, question_data["frage"])
            if new_questions:
                generated_questions.extend(new_questions)
            time.sleep(sleep_after_generation)

        # Check if it's time to flush QA results
        if (datetime.now() - last_qa_flush).total_seconds() >= qa_flush_interval * 60 and qa_results_buffer:
            job_config.output_file, volume = save_qa_result(job_config.output_file, qa_results_buffer, volume, max_output_file_size_gb)
            qa_results_buffer = []
            last_qa_flush = datetime.now()
            log_message("Current QA successfully flushed", job_config.job_id, "info")
        
        # Check if it's time to flush generated questions
        if (datetime.now() - last_questions_flush).total_seconds() >= questions_flush_interval * 60:
            generated_questions, last_questions_flush = flush_questions(job_config.input_file, questions, generated_questions)
        
        log_message("Current question successfully processed", job_config.job_id, "info")

    job_config.output_file, volume = save_qa_result(job_config.output_file, qa_results_buffer, volume, max_output_file_size_gb)
    #flush_questions(job_config.input_file, questions, generated_questions)

def load_job_configs(config_file: str) -> List[JobConfig]:
    """Load job configurations from the JSON config file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return [JobConfig(**job) for job in config['jobs'] if job.get('enabled', True)]



def main():
    """Main function to process multiple jobs in parallel."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process questions using hybrid search with multiple jobs')
    parser.add_argument('--config', required=True, help='Path to the job configuration file')
    args = parser.parse_args()
    
    # Load job configurations
    job_configs = load_job_configs(args.config)
    
    if not job_configs:
        logging.error("No enabled jobs found in configuration")
        return
           
    # Create a list of arguments for each job
    job_args = [(config) for config in job_configs]
        
    # Create a process pool with the number of jobs
    with Pool(processes=len(job_configs)) as pool:
        try:
            # Start processing jobs in parallel
            pool.map(process_job, job_args)
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, stopping jobs...")
            pool.terminate()
            pool.join()
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error processing jobs: {str(e)}")
            pool.terminate()
            pool.join()
            sys.exit(1)
            
    logging.info("All jobs completed successfully")

if __name__ == "__main__":
    main()
