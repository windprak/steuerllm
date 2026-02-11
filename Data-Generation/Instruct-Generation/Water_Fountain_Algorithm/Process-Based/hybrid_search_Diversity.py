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
    if fragetyp == "Klassifikationsaufgaben":
        return (
            "Du wirst gebeten, steuerrechtliche Aussagen als wahr oder falsch zu klassifizieren oder sie einer bestimmten Kategorie zuzuordnen. "
            "Bei mehrklassigen Klassifikationen sollst du die verschiedenen Kategorien (z. B. Einkunftsarten) korrekt erkennen und zuordnen. "
            "Stelle sicher, dass deine Klassifikationen konsistent und logisch sind und auf den geltenden steuerrechtlichen Vorschriften basieren. "
            "Beispiel: 'Einnahmen aus Vermietung und Verpachtung sind steuerfrei.' -> 'Falsch'."
        )
    elif fragetyp == "Lückentextaufgaben":
        return (
            "Du wirst mit steuerrechtlichen Sachverhalten konfrontiert, bei denen du bestimmte Lücken mit den richtigen Begriffen oder Informationen füllen musst. "
            "Achte darauf, dass die Lücken logisch sind und im Kontext des Sachverhalts passen. "
            "Die geforderten Begriffe sollten präzise und korrekt in Übereinstimmung mit den relevanten steuerrechtlichen Regelungen verwendet werden. "
            "Beispiel: 'Einnahmen aus ________ gehören zu den Überschusseinkunftsarten.' -> 'Vermietung und Verpachtung'."
        )
    elif fragetyp == "Reihenfolgeaufgaben":
        return (
            "Deine Aufgabe ist es, die Schritte eines steuerrechtlichen Verfahrens oder Prozesses in der richtigen Reihenfolge anzuordnen. "
            "Achte darauf, dass die Reihenfolge korrekt ist und mit den tatsächlichen Abläufen im Steuerrecht übereinstimmt. "
            "Vermeide es, Schritte zu vertauschen oder wichtige Schritte zu übersehen. "
            "Beispiel: 'Bringe die Schritte der Einkommensteuerveranlagung in die richtige Reihenfolge: (1) Steuerbescheid erlassen, (2) Steuererklärung prüfen, (3) Bescheinigung der Steuerpflicht anfordern.' -> '(3), (2), (1)'."
        )
    elif fragetyp == "Textverständnisaufgaben":
        return (
            "Lese den gegebenen Text oder Gesetzesauszug sorgfältig und beantworte Fragen, die sich auf die Schlüsselinformationen des Textes beziehen. "
            "Deine Antworten sollten präzise und klar sein, basierend auf einer gründlichen Analyse des Textes. "
            "Extrahiere die relevanten Informationen und achte darauf, keine irrelevanten Details zu verwenden. "
            "Beispiel: (Gesetzestext zur Einkunftsart 'Gewerbebetrieb') Frage: 'Welche Kriterien müssen erfüllt sein, damit Einkünfte aus einem Gewerbebetrieb vorliegen?' -> 'Selbstständige, nachhaltige Betätigung mit Gewinnerzielungsabsicht, die sich am allgemeinen wirtschaftlichen Verkehr beteiligt.'"
        )
    elif fragetyp == "Argumentationsaufgaben":
        return (
            "Du wirst gebeten, eine steuerrechtliche Argumentation zu entwickeln, die auf relevanten Vorschriften basiert. "
            "Die Argumentation sollte gut strukturiert sein, klar und überzeugend, mit einer logischen Erklärung, warum ein bestimmter steuerlicher Sachverhalt so behandelt wird. "
            "Berücksichtige alle relevanten steuerrechtlichen Aspekte und stelle sicher, dass dein Argumentationsprozess nachvollziehbar ist. "
            "Beispiel: 'Sollten Betriebsausgaben in voller Höhe abzugsfähig sein? Begründe.' -> 'Ja, weil sie unmittelbar der Erzielung von Einkünften dienen, gemäß § 4 Abs. 4 EStG.'"
        )
    elif fragetyp == "Matching-Aufgaben":
        return (
            "Du musst Begriffe und deren zugehörige Definitionen oder Regelungen korrekt zuordnen. "
            "Achte darauf, dass jede Zuordnung präzise ist und keine Missverständnisse in der rechtlichen Interpretation auftreten. "
            "Jede Zuordnung sollte mit den entsprechenden gesetzlichen Vorschriften übereinstimmen. "
            "Beispiel: 'Ordne die Einkunftsarten (A) bis (F) der korrekten Kategorie (1-3) zu: A. Gewerbebetrieb, B. Kapitalvermögen, C. Vermietung und Verpachtung' -> 'A-2, B-1, C-1'."
        )
    elif fragetyp == "Fallstudien / Szenario-basierte Aufgaben":
        return (
            "Löse komplexe steuerrechtliche Szenarien, indem du den Sachverhalt analysierst und die steuerlichen Auswirkungen erklärst. "
            "Berücksichtige alle relevanten steuerrechtlichen Vorschriften und beantworte die Fragen detailliert. "
            "Achte darauf, dass deine Antworten alle relevanten Aspekte abdecken und die steuerliche Behandlung des Falls korrekt dargestellt wird. "
            "Beispiel: 'Herr Müller erzielt Einkünfte aus Gewerbebetrieb (50.000 €) und Kapitalvermögen (10.000 €). Wie wird die Einkommensteuer berechnet?' -> 'Die Einkünfte aus Gewerbebetrieb werden nach § 15 EStG behandelt, die aus Kapitalvermögen gemäß § 20 EStG. Beide Einkünfte fließen in das zu versteuernde Einkommen ein.'"
        )
    elif fragetyp == "Ergänzungsaufgaben":
        return (
            "Vervollständige unvollständige Informationen oder Szenarien, indem du die fehlenden Teile logisch und präzise ergänzt. "
            "Deine Ergänzungen sollten sinnvoll im Kontext des gegebenen Sachverhalts sein und auf den relevanten steuerrechtlichen Regelungen basieren. "
            "Erkläre jede Ergänzung gründlich und sortiere sie nach ihrer Wichtigkeit. "
            "Beispiel: 'Das zu versteuernde Einkommen setzt sich aus 1) Einkünften aus nichtselbstständiger Arbeit, 2) Einkünften aus Gewerbebetrieb und 3) _______ zusammen.' -> 'Einkünften aus Kapitalvermögen.'"
        )
    elif fragetyp == "Erklärungsaufgaben":
        return (
            "Erkläre steuerrechtliche Begriffe oder Konzepte in einfacher und verständlicher Sprache. "
            "Verwende Beispiele oder Vergleiche zur Verdeutlichung und stelle sicher, dass deine Erklärung detailliert ist. "
            "Die Erklärung sollte für Personen ohne tiefgehendes steuerrechtliches Wissen verständlich sein. "
            "Beispiel: 'Erkläre den Unterschied zwischen Einnahmen-Überschuss-Rechnung und Bilanzierung.' -> 'Die Einnahmen-Überschuss-Rechnung ist eine vereinfachte Methode zur Gewinnermittlung nach § 4 Abs. 3 EStG.'"
        )
    elif fragetyp == "Rechnungsaufgaben":
        return (
            "Führe steuerrechtliche Berechnungen präzise durch und präsentiere die Ergebnisse korrekt. "
            "Stelle sicher, dass alle Rechenschritte klar und nachvollziehbar sind und auf den entsprechenden steuerrechtlichen Vorschriften basieren. "
            "Achte darauf, dass deine Berechnungen korrekt durchgeführt werden und keine Fehler enthalten. "
            "Beispiel: 'Berechne die Steuer auf Kapitalerträge: Kapitalertrag = 10.000 €, Abgeltungsteuersatz = 25%.' -> 'Die Steuer beträgt 2.500 €.'"
        )
    elif fragetyp == "Regel-Extraktionsaufgaben":
        return (
            "Extrahiere relevante steuerrechtliche Regeln oder Paragraphen aus einer Beschreibung oder einem Sachverhalt. "
            "Die extrahierten Regeln sollten präzise und relevant sein, und du solltest die Quelle der Regel angeben. "
            "Erkläre, warum diese Regel im Kontext des Sachverhalts wichtig ist. "
            "Beispiel: 'Nenne die Paragraphen, die die steuerliche Behandlung von Werbungskosten regeln.' -> '§ 9 EStG.'"
        )
    elif fragetyp == "Vergleichsaufgaben":
        return (
            "Vergleiche steuerrechtliche Sachverhalte oder Regelungen und erläutere die Unterschiede und Gemeinsamkeiten. "
            "Deine Antwort sollte die wichtigsten Aspekte der beiden verglichenen Sachverhalte klar und detailliert darstellen. "
            "Vergleiche die steuerlichen Auswirkungen und stelle sicher, dass du alle relevanten Unterschiede und Gemeinsamkeiten berücksichtigst. "
            "Beispiel: 'Vergleiche die steuerliche Behandlung von Einnahmen aus Vermietung und Verpachtung mit Einnahmen aus Gewerbebetrieb.' -> 'Einnahmen aus Vermietung und Verpachtung werden gemäß § 21 EStG versteuert, während Einnahmen aus Gewerbebetrieb nach § 15 EStG behandelt werden...'"
        )
    elif fragetyp == "Generierung neuer Aufgaben":
        return (
            "Erstelle eine plausible und praxisnahe Prüfungsfrage zu einem gegebenen Paragraphen. "
            "Die Frage sollte auf dem Paragraphen basieren und realistische steuerrechtliche Szenarien abdecken. "
            "Achte darauf, dass die Frage herausfordernd ist und tiefgehendes steuerrechtliches Wissen erfordert. "
            "Beispiel: Wenn die Frage ist 'Erstelle eine Frage zu § 15 EStG.' dann antworte mit einer Frage wie z.B. 'Welche Kriterien müssen erfüllt sein, damit Einkünfte aus Gewerbebetrieb vorliegen?'"
            "Wichtig: Du musst eine Frage erstellen, welche für Klausuren oder Examen benutzt werden kann und nicht den Artikel grunlegend erklären"
        )
    elif fragetyp == "Übersetzungsaufgaben":
        return (
            "Übersetze steuerrechtliche Begriffe oder Paragraphen in eine einfachere, verständliche Sprache. "
            "Die Übersetzung sollte präzise und klar sein und komplexe juristische Sprache in eine zugängliche Form bringen. "
            "Stelle sicher, dass die Kernaussage des Paragraphen oder Begriffs erhalten bleibt und verständlich erklärt wird. "
            "Beispiel: 'Erkläre § 19 EStG in einfacher Sprache.' -> 'In § 19 EStG geht es um Einkünfte aus nichtselbstständiger Arbeit. Das bedeutet, was man durch einen Job als Arbeitnehmer verdient.'"
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
            {"role": "system", "content": "Du bist ein hilfreicher Assistent für juristische Fragen und Aufgaben. Beantworte die Frage/Aufgabe basierend auf dem gegebenen Kontext. Wenn du dir bei der Antwort nicht sicher bist, gib dies an. Vermeide Aussagen, wie 'basierend auf gegebenem Kontext' oder 'aus dem gegebenem Text'. Wichtig ist, dass du sofern möglich wichtige Paragraphen, Rechtstexte oder Urteile zitierst."},
            {"role": "user", "content": f"Kontext:\n{context}\n\nFrage/Aufgabe: {query}\n\nQualitätskriterien: {qualität}.\n\nEs ist sehr wichtig das du die genau die Frage/Aufgabe beantwortest und auf die Qualitätskriterien achtest!\n\n Wenn die Antwort nicht aus dem Kontext hervorgeht, antworte nur mit: 'Es tut mir leid, aber es der Kontext ist nicht ausreichend.'. Zitiere wenn möglich Rechtstexte, Paragraphen, Gerichtsurteile oder andere staatliche offiziele Quellen. Nutze nur Quellen die sich explizit auf Deutschland beziehen, nutze keine Quellen aus Österreich oder der Schweiz."}
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
    Du bist ein Experte im deutschen Steuerrecht. Deine Aufgabe ist es, qualitativ hochwertige und sinnvolle Aufgaben zu erstellen, die sich auf den gegebenen Kontext und die Antwort beziehen. Deine Aufgabe ist es, ein spezifischeres bzw. anderes steuerrechtliches Gebiet als die alte Aufgabe zu bearbeiten. Das heißt, du nimmst den gegebenen Kontext und versuchst, einen oder mehrere detaillierte oder andere Bereiche des deutschen Steuerrechts zu identifizieren. Über diese Bereiche erstellst du dann die Aufgaben.
    Jede Aufgabe muss zu einem der folgenden vordefinierten Typen gehören. Generieren Sie nur Aufgaben, die zum Kontext passen. 
    Beachten Sie die detaillierten Beschreibungen und Beispiele und wichtigen Aspekte für jeden Typ:

    1. Klassifikationsaufgaben
        Klassifikationsaufgaben sollten klare, prägnante Aussagen enthalten, die überprüfbar sind. Dabei ist darauf zu achten, dass die Aussage entweder eine klare Richtigkeit oder Falschheit besitzt, ohne Interpretationsspielraum. Bei Mehrklassenklassifikationen sollte jede Option eindeutig zuordenbar sein. Formulierungen sollten relevante steuerrechtliche Begriffe enthalten und die Aussagen auf konkrete, im Steuerrecht geregelte Sachverhalte beziehen.
        Wichtige Aspekte:
        Aussagen müssen spezifisch, aber nicht trivial sein.
        Klare Trennung zwischen Fakten und Meinungen.
        Verweise auf Paragraphen, die relevant sind, erhöhen die Fachlichkeit.
        Beispiel:
        "Stimmt es, dass Einnahmen aus Vermietung und Verpachtung gemäß § 21 EStG stets steuerfrei sind? Klassifizieren Sie die Aussage als wahr oder falsch."
        "Ordnen Sie die folgenden Einkünfte der richtigen Einkunftsart gemäß § 2 Abs. 1 EStG zu: Einnahmen aus dem Verkauf landwirtschaftlicher Erzeugnisse, Dividenden aus einer Aktienbeteiligung und Einkünfte aus der Verpachtung einer Ferienwohnung."
    
    2. Lückentextaufgaben
        Die Lücken sollten auf wesentliche steuerrechtliche Informationen abzielen, die für das Verständnis von Gesetzen oder Sachverhalten unerlässlich sind. Die fehlenden Begriffe oder Zahlen sollten direkt aus dem Gesetz ableitbar sein oder typischen Fallbeispielen entsprechen. Der Kontext um die Lücke muss genügend Hinweise bieten, um die korrekte Lösung nachvollziehbar zu machen.
        Wichtige Aspekte:
        Die Lücken müssen eindeutig und nicht mehrdeutig ausfüllbar sein.
        Der Satzbau soll die Bedeutung der Lücke klar hervorheben.
        Bezug auf praxisrelevante Steuerregelungen oder Prozesse.
        Beispiel:
        "Einnahmen aus ________ gehören zu den Überschusseinkunftsarten."
    
    3. Reihenfolgeaufgaben
        Hier sollten die Schritte eines steuerlichen Prozesses so beschrieben werden, dass sie in einer logischen Reihenfolge angeordnet werden können. Die Schritte müssen fachlich korrekt und vollständig sein, ohne dass unnötige oder ablenkende Elemente eingefügt werden.
        Wichtige Aspekte:
        Schritte klar und präzise formulieren.
        Reihenfolge sollte nachvollziehbar und für den Steueralltag relevant sein.
        Es dürfen keine Mehrdeutigkeiten in der Bedeutung der Schritte auftreten.
        Beispiel:
        "Bringe die Schritte der Einkommensteuerveranlagung in die richtige Reihenfolge: (1) Steuerbescheid erlassen, (2) Steuererklärung prüfen, (3) Bescheinigung der Steuerpflicht anfordern."
    
    4. Textverständnisaufgaben
        Die Aufgaben sollten auf einem gegebenen Text basieren, der steuerrechtliche Sachverhalte oder Gesetzesauszüge enthält. Die Frage muss gezielt auf spezifische Informationen im Text abzielen, sodass der Text analysiert und interpretiert werden muss. Der Text sollte nicht zu lang sein, um die Fokussierung zu erleichtern, aber dennoch ausreichend komplex, um fundierte Antworten zu ermöglichen.
        Wichtige Aspekte:
        Die Aufgaben müssen den Text klar referenzieren und logisch darauf aufbauen.
        Relevanz zum Steuerrecht sicherstellen (z. B. durch Paragraphen oder typische Fallkonstellationen).
        Der Text sollte keine missverständlichen oder überflüssigen Informationen enthalten.
        Beispiel:
        [Füge hier den Gesetzestext zur Einkunftsart "Gewerbebetrieb" ein], dann "Nenne die Kriterien, die erfüllt sein müssen, damit Einkünfte aus einem Gewerbebetrieb vorliegen."
    
    5. Argumentationsaufgaben
        Die Aufgaben sollten einen steuerrechtlichen Sachverhalt oder eine Regelung enthalten, zu der ein Standpunkt erarbeitet werden muss. Dabei sollte die Fragestellung genügend Kontext bieten, um fundierte Argumente entwickeln zu können. Die Szenarien sollten realistisch und relevant sein, mit einer klaren Aufforderung, steuerrechtliche Regelungen oder Grundsätze zu berücksichtigen.
        Wichtige Aspekte:
        Realistische, praxisbezogene Szenarien nutzen.
        Offene Fragestellung, die verschiedene Perspektiven zulässt.
        Bezug zu Gesetzesgrundlagen oder Steuerprinzipien herstellen.
        Beispiel:
        "Begründe warum Betriebsausgaben in voller Höhe abzugsfähig sein sollten."
    
    6. Matching-Aufgaben
        Steuerrechtliche Begriffe oder Konzepte sollten klar definiert werden und einer passenden Kategorie, Vorschrift oder Bedeutung zugeordnet werden können. Die Zuordnungen müssen eindeutig sein, ohne dass Begriffe mehrfach zutreffen können.
        Wichtige Aspekte:
        Begriffe und Kategorien sollten fachlich korrekt und direkt zuordenbar sein.
        Die Kategorien sollten ausreichend differenziert sein, um die Zuordnung anspruchsvoll zu machen.
        Es sollten möglichst gängige steuerrechtliche Begriffe verwendet werden.
        Beispiel:
        "Ordnen Sie die folgenden Einkunftsarten den korrekten Kategorien zu: (1) Gewinneinkünfte, (2) Überschusseinkünfte und (3) nicht steuerbare Einkünfte."

    7. Ergänzungsaufgaben
        Ergänzungsaufgaben sollten Sachverhalte oder Gesetzestexte enthalten, bei denen gezielt ein Detail fehlt, das sinnvoll ergänzt werden muss. Die fehlenden Informationen sollten dabei klar definierbar sein, sodass keine Mehrdeutigkeit bei der Antwort entsteht. Wichtig ist, dass der Kontext genug Hinweise bietet, um die Lücke korrekt zu ergänzen.
        Wichtige Aspekte:
        Die Lücken müssen präzise und eindeutig zu ergänzen sein.
        Der Kontext sollte logisch und vollständig sein, um die Lösung zu ermöglichen.
        Inhalte sollten praxisrelevant oder direkt aus dem Gesetz ableitbar sein.
        Beispiel:
        "Nach § 2 Abs. 1 EStG unterscheidet das deutsche Steuerrecht sieben Einkunftsarten. Dazu gehören: 1) Einkünfte aus Land- und Forstwirtschaft, 2) Einkünfte aus Gewerbebetrieb, 3) Einkünfte aus selbstständiger Arbeit, 4) Einkünfte aus ________, 5) Einkünfte aus Kapitalvermögen, 6) Einkünfte aus Vermietung und Verpachtung und 7) sonstige Einkünfte im Sinne des § 22 EStG. Ergänzen Sie die fehlende Einkunftsart."
    
    8. Erklärungsaufgaben
        Erklärungsaufgaben sollten sich auf steuerrechtliche Begriffe, Konzepte oder Regelungen beziehen, die vom LLM klar und verständlich erläutert werden können. Die Aufgabenstellung sollte präzise formuliert sein und deutlich machen, welches Konzept oder welcher Begriff erklärt werden soll. Fachlichkeit und Verständlichkeit müssen dabei gleichermaßen gewährleistet sein. Es soll nicht in Form von Fragen oder Szenarien gestellt werden.
        Wichtige Aspekte:
        Die Begriffe oder Konzepte sollten praxisrelevant und fachlich korrekt sein.
        Die Aufgabenstellung sollte einen klaren Fokus haben.
        Die Frage sollte eine umfassende, aber nicht ausschweifende Antwort ermöglichen.
        Beispiel:
        "Erkläre den Unterschied zwischen Einnahmen-Überschuss-Rechnung und Bilanzierung."
    
    9. Regel-Extraktionsaufgaben
        Regel-Extraktionsaufgaben sollten sich auf Gesetzestexte oder steuerrechtliche Sachverhalte beziehen, aus denen spezifische Regeln, Paragraphen oder Vorgaben extrahiert werden müssen. Der Fokus liegt darauf, dass die relevanten Informationen im Text identifiziert und korrekt zugeordnet werden.
        Wichtige Aspekte:
        Der Text oder Kontext muss klar definieren, welche Informationen extrahiert werden sollen.
        Die Regeln oder Paragraphen sollten spezifisch und direkt aus dem Text ableitbar sein.
        Fachliche Präzision ist essenziell.
        Beispiel:
        "Welche Voraussetzungen müssen laut § 10b EStG erfüllt sein, damit eine Spende steuerlich abgesetzt werden kann?"
    
    10. Vergleichsaufgaben
        Vergleichsaufgaben sollten zwei oder mehr steuerliche Sachverhalte oder Regelungen enthalten, deren Unterschiede und Gemeinsamkeiten analysiert werden müssen. Die Fragestellung sollte klar formuliert sein und darauf abzielen, dass die relevanten Aspekte systematisch gegenübergestellt werden.
        Wichtige Aspekte:
        Sachverhalte oder Regelungen müssen vergleichbar sein.
        Die Fragestellung sollte eindeutig und auf den Vergleich fokussiert sein.
        Der Bezug zu Paragraphen oder steuerrechtlichen Grundlagen sollte klar ersichtlich sein.
        Beispiel:
        "Vergleiche die steuerliche Behandlung von Einnahmen aus Vermietung und Verpachtung mit Einnahmen aus Gewerbebetrieb."
        
    11. Generierung neuer Aufgaben
        Bei dieser Aufgabenkategorie sollte das LLM auf Basis eines gegebenen Paragraphen oder steuerrechtlichen Themas eigenständig plausible und fachlich korrekte Fragen entwickeln. Die Aufgaben sollten praxisrelevante Themen abdecken und in ihrer Komplexität an typische Prüfungsfragen angepasst sein.
        Wichtige Aspekte:
        Der Paragraph oder das Thema sollte eindeutig vorgegeben werden.
        Die generierte Frage muss den Anforderungen an fachliche Präzision und Relevanz genügen.
        Die Frage sollte nicht trivial sein und Steuerwissen herausfordern.
        Beispiel:
        "Erstelle eine Frage zu § 15 EStG."
    
    12. Übersetzungsaufgaben
        Übersetzungsaufgaben sollten darauf abzielen, komplexe steuerrechtliche Formulierungen in eine verständliche Sprache umzuwandeln. Die Fragestellung sollte präzise formulieren, welcher Teil des Gesetzes oder Textes vereinfacht werden soll, und eine Zielgruppe berücksichtigen, z. B. Laien oder Steuerfachleute.
        Wichtige Aspekte:
        Die Ausgangstexte müssen fachlich korrekt sein.
        Der Fokus sollte auf einer präzisen, aber leicht verständlichen Sprache liegen.
        Die Zielgruppe (z. B. Laien) sollte bei der Vereinfachung berücksichtigt werden.
        Beispiel:
        "Erkläre § 19 EStG in einfacher Sprache."

    **Generelle Anforderungen:**
    - Generiere bis zu {max_questions} Aufgaben (mindestens eine).
    - Wähle nur die Aufgabentypen aus, die für den gegebenen Kontext wirklich sinnvoll sind.
    - Formatiere die Ausgabe als JSON-Array mit den Feldern 'frage', 'typ' und 'quelle'.
    - Lasse das Feld 'quelle' immer leer.
    - Die Aufgaben sollen ausschließlich auf Deutsch geschrieben werden.
    - Die Aufgaben sollen der ursprünglichen Aufgabe nicht ähnlich sein.
    - Nutze niemals die Beispielaufgaben, sondern lerne von ihnen.
    - Die Aufgaben sollen so ausführlich, wie möglich sein und wenn du Beispiele oder Szenarien nennst, sollten diese überaus detailiert sein.
    - Die Aufgaben sollen auf kreativen Weg formuliert werden, um den Kontext zu verstehen und die Aufgaben zu vermeiden, die bereits in der Antwort enthalten sind.
    """

    # User prompt specifying the task
    user_prompt = f"""
    Context: {context_text}
    Old question: {question}
    
    Generate up to {max_questions} tasks, the amount of questions depends on the context and if there is a possibility to create a task. The tasks should be high-quality tax-related tasks based on the context. Check for each task type, if there is a possibility to generate a task, if you are unsure skip the task type as you explicitly do not need to create exactly {max_questions} tasks.
    If possible, the tasks should dive into another topic or a more specified field as the old task.
    Be very specific with the tax and law related terms and if suitable incorperate them in the tasks.
    IMPORTANT: Do not use the same task multiple times, also watch out that you do not create the same task multiple times which simply uses different terms.
    If possible, all tasks should have a different tax law topic and should not address the same issue.
    The tasks should not be phrased as a question. Be creative in your question generation, you can use synonyms, e.g. "Erstellen -> Kreieren" or "Erkläre -> Erläutere/Definiere/Führe aus etc." 
    Only include tasks that fit the context. Format your response as a JSON array, as shown below:
    [
        {{"frage": "Example task?", "typ": "Type of the task (e.g. "Vergleichsaufgaben")", "quelle": "Should be empty always"}},
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
                "top_p": 0.9
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
    #generated_questions = []
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
    flush_questions(job_config.input_file, questions, generated_questions)

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
