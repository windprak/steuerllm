from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import random

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
            self.flush_interval = config.get('flush_interval', 180)  # Flush interval in seconds
            self.model = config['model']
            self.max_questions = config['amount_questions']
            self.checkpoint_file = "checkpoint_file.json"
            self.reps = config.get('reps', 1)

    def load_processed_articles(self, log_path: str) -> set:
        """Lade die IDs der bereits verarbeiteten Artikel aus der Log-Datei."""
        if not os.path.exists(log_path):
            return set()
        with open(log_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)

    def log_processed_articles(self, log_path: str, processed_ids):
        """Füge eine verarbeitete Artikel-ID in die Log-Datei ein."""
        with open(log_path, 'a', encoding='utf-8') as f:
            for article_id in processed_ids:
                f.write(f"{article_id}\n")



    def process(self, log_path="processed_Pre.log"): # Here is logs which IDs already have been processed
        self.processed_ids = self.load_processed_articles(log_path)
        results = []
        processed_ids = []
        start_time = time.time()

        # Iterate through each file
        data_path = Path(self.input_file)
        for file_path in data_path.glob("*.json"):
            logger.info(f"Processing file: {file_path}")
            articles = self.load_articles(file_path)

            # Split the articles among the endpoints
            articles_per_endpoint = self.split_articles_on_endpoints(articles)

            with ThreadPoolExecutor(max_workers=self.num_jobs) as executor:
                # Create future objects for the tasks
                futures = {executor.submit(self.call_endpoint, article): article for article in articles_per_endpoint}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        # Iterate over the results if it is a list
                        if isinstance(result, list):
                            if result:
                                for res in result:  # Add each individual result to the list
                                    results.append(res)
                                    processed_ids.append(result[0]['id'])
                                    logger.info("Current Question successfully processed")
                        else:
                            logger.error(f"Unexpected result format: {result}")
                          # Assuming result is a list of dicts
                        
                    except Exception as e:
                        logger.error(f"Error processing article: {futures[future]}. Error: {e}")

                    # Periodic flush
                    if time.time() - start_time >= self.flush_interval:
                        self.flush_results(results, mode='a')
                        self.log_processed_articles(log_path, processed_ids)
                        results.clear()
                        processed_ids.clear()
                        start_time = time.time()

            # Flush the remaining results for the current file
            if results:
                self.flush_results(results, mode='a')
                self.log_processed_articles(log_path, processed_ids)
                results.clear()
                processed_ids.clear()

        logger.info("All files processed successfully.")

    
    def load_articles(self, input_path: str) -> List[Dict]:
        articles = []
        data_path = Path(input_path)
        
        # Check if the path is a file or directory
        if data_path.is_dir():  # Directory case
            files_to_process = data_path.glob("*.json")
        elif data_path.is_file():  # Single file case
            files_to_process = [data_path]
        else:
            raise ValueError("The provided input path is neither a valid file nor a directory.")
        
        # Process each file
        for file_path in files_to_process:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for idx, obj in enumerate(data):
                    article_id = f"article_{file_path.stem}_{idx}"

                    # Wenn die ID bereits verarbeitet wurde, überspringe den Artikel
                    if article_id in self.processed_ids:
                        continue

                    sources = obj.get("sources", [])
                    searxng_sources = [source for source in sources if source.get("source_info", {}).get("source") == "SearXNG"]
                    
                    if len(searxng_sources) > 2:
                        article_text = "\n".join([source["text"] for source in searxng_sources])
                        articles.append({
                            "article": article_text,
                            "id": article_id
                        })

                        # Mark the article as processed
                        self.processed_ids.add(article_id)

        print("Articles loaded")
        return articles


    def split_articles_on_endpoints(self, articles: List[str]) -> List[Dict]:
        """Distributes the items evenly among the endpoints."""
        endpoint_articles = []
        num_endpoints = len(self.endpoints)
        for i, article in enumerate(articles):
            # Determines the endpoint for this item
            endpoint_index = i % num_endpoints
            endpoint_articles.append({'article': article, 'endpoint': self.endpoints[endpoint_index]})
        return endpoint_articles



    def call_endpoint(self, article_data: Dict) -> List[Dict]:
        """Call FAU endpoint to generate three question-answer pairs."""
        article = article_data['article']
        article_text = article_data['article']['article']
        endpoint = article_data['endpoint']
        results = []

        try:
            for _ in range(self.reps):  
                payload = self.prepare_payload_q(article_text)
                question_response = requests.post(endpoint, json=payload, timeout=600)
                question_response.raise_for_status()

                # Process the answer for the question
                question_content = question_response.json()["choices"][0]["message"]["content"]
                cleaned_question = question_content.strip()

                if cleaned_question == "Es tut mir Leid, aber der Kontext ist nicht ausreichend.":
                    break

                payload_a = self.prepare_payload_a(article, cleaned_question)
                answer_response = requests.post(endpoint, json=payload_a, timeout=600)
                answer_response.raise_for_status()

                # Process the answer for the answer
                answer_content = answer_response.json()["choices"][0]["message"]["content"]
                cleaned_answer = answer_content.strip()

                results.append({
                    'question': cleaned_question,
                    'answer': cleaned_answer,
                    'id': article['id']
                })

                user_followup = f"""
                Jetzt erstelle bitte eine weitere Frage auf den gleichen Kontext wie gerade eben, die thematisch anders ist als die vorherige Frage. Es ist ganz wichtig, dass du nicht mit einer ähnlichen Frage antwortest, falls du keine andere Frage als die ursprüngliche stellen kannst musst du unbedingt mit "Es tut mir Leid, aber der Kontext ist nicht ausreichend." antworten.
                Eine Frage gilt auch als ähnlich, wenn du ein paar Details an der Story änderst, es geht darum dass du die Aussage der Frage grundlegend änderst. Ändere also bitte nicht nur ein paar Details an der Story, sondern gehe in unterschiedliche Thematische Richtungen oder Fragestellungen.
                Die neue Frage soll wenn sie beantwortet wird eine andere Antwort haben als die letzte Frage. Du kannst auch unterschiedliche Fragetypen nehmen.
                """
                
                payload["messages"].append({"role": "assistant", "content": cleaned_question})
                payload["messages"].append({"role": "user", "content": user_followup})


            return results

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if payload and payload_a:
                return self.call_endpoint_rec(article, payload, payload_a=payload_a, tried_endpoints=[endpoint])
            else:
                print("Request ended before creating payloads")


    def call_endpoint_rec(self, article, payload_q, payload_a, tried_endpoints: List[str]) -> List[Dict]:
        """Recursive method to retry with different endpoints for multiple pairs."""
        available_endpoints = [ep for ep in self.endpoints if ep not in tried_endpoints]
        if not available_endpoints:
            return [{'question': "Failed!", 'answer': 'All endpoints failed', 'id': article['id']}]

        endpoint = random.choice(available_endpoints)
        tried_endpoints.append(endpoint)
        results = []

        try:
  
            question_response = requests.post(endpoint, json=payload_q, timeout=600)
            question_response.raise_for_status()

            question_content = question_response.json()["choices"][0]["message"]["content"]
            cleaned_question = question_content.strip()


            if cleaned_question != "Es tut mir Leid, aber der Kontext ist nicht ausreichend.":
                answer_response = requests.post(endpoint, json=payload_a, timeout=600)
                answer_response.raise_for_status()

                answer_content = answer_response.json()["choices"][0]["message"]["content"]
                cleaned_answer = answer_content.strip()
            else:
                cleaned_answer = "Es tut mir Leid, aber der Kontext ist nicht ausreichend."

            results.append({
                'question': cleaned_question,
                'answer': cleaned_answer,
                'id': article['id']
            })

            return results

        except requests.exceptions.RequestException as e:
            print(f"Request failed for endpoint {endpoint}: {e}")
            return self.call_endpoint_rec(article, payload_q, payload_a, tried_endpoints)

    def prepare_payload_q(self, article: str) -> Dict:
        """Prepare the request payload for the FAU endpoint."""
        system_prompt = """
    You are an expert in German tax law. Your task is to generate a high-quality and meaningful question related to the given context and answer. This means you take the given context and try to identify a detailled area of german tax law.
    The question must belong to one of the following six predefined types. Only generate a question that is relevant to the context. 
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
        - "Ich habe die Gebäudeversicherung für März 2023 - Feb 2024 im März überwiesen (12000 EUR). Ich bin nach § 4 Abs. 1 EStG / § 5 Abs. 1 EStG bilanzierungspflichtig. Bilde mir den Buchungssatz!"
    - Qualitätskriterien: 
        - Präzise Fragestellung zum Buchungssatz.


    **3. Rahmenfragen**
    - Ziel: Ermittlung von wichtigen Aspekten, die bei einem spezifischen steuerrechtlichen Kontext berücksichtigt werden müssen.
    - Beispiele:
        - "Meine GmbH hält 100 Aktien der ABC-AG, welche sie zu 100 EUR jeweils gekauft hat. Erläutere mir die Besteuerungsfolgen, wenn die Aktien zu 200 EUR jeweils verkauft werden. Mache eine konkrete Berechnung, wie viel Körperschaftsteuer ich bezahlen muss."
        - "A ist Gesellschafter an der B-GmbH und nimmt sich aus der Vermögen der B-GmBH einen Laptop, welcher einen gemeinen Wert von 1.500 EUR hat. Ein Kaufvertrag liegt nicht vor, eine Kompensation wird nicht gewährt. Welche Rechtsfolgen resultieren daraus. Berechne, wie sich die Steuerlasten in der Körperschaftsteuer verändern"
    - Qualitätskriterien:
        - Strukturiert dargestellter Sachverhalt mit einer kreativen Fragestellung.
        - Fiktive Beispiele oder praktische Szenarien.
        - Frage nach dem steuerlichen Rahmen oder der steuerlichen Verhaltung (kreativ formuliert).
        - Szenario mindestens 2 Sätze lang

    **4. Korrekturfragen**
    - Ziel: Überprüfung einer vorgeschlagenen Lösung. Die Frage soll klären, ob die Lösung korrekt ist oder nicht, und bei Bedarf die Begründung liefern.
    - Beispiel:
        - "Lara Baumann und Tim Reinders sind begeisterte Jogger, die gemeinsam einen Lauf-Shop in Hannover eröffnen möchten. Ihr Unternehmen “Der Lauf-Partner” soll Kunden helfen, ihre Ziele im Bereich Fitness und Gesundheit zu erreichen. Im Rahmen ihrer Geschäftstätigkeit haben sie folgende Vereinbarungen getroffen. Bitte entscheiden Sie, ob diese Abmachungen gültig, anfechtbar oder nichtig sind. Wenn ein Rechtsgeschäft Ihrer Meinung nach nichtig oder anfechtbar ist, bitte geben Sie den relevanten Grund für die Nichtigkeit oder Anfechtbarkeit an. Im Falle eines gültigen Rechtsgeschäfts brauchen Sie keine Begründung anzugeben. Der Lauf-Shop hat einem Kunden ein neues Laufband verkauft, das als fabrikneu zum Neupreis angeboten wurde. Es sei jedoch bemerkt, dass das Gerät bereits auf einer Messe als Vorführgerät eingesetzt war."
    - Qualitätskriterien:
        - Klare Möglichkeit zur booleschen Bewertung ("Ja/Nein").
        - Strukturiert dargestellter Sachverhalt mit einer kreativen Fragestellung.
        - Fiktive Beispiele oder praktische Szenarien.
        - Szenario mindestens 2 Sätze lang

    **5. Quellenabfragen**
    - Ziel: Fragen sollen Gesetzestexte, Paragraphen oder andere Quellen für ein Problem identifizieren.
    - Beispiele:
        - "Mein Mandant ist 55 Jahre alt und hält Anteile an einer erfolgreichen GmbH. Mit 65 möchte er spätestens die Anteile an einen Konkurrenten verkaufen. Überlege dir eine Gestaltung, wie die Steuerlast maximal reduziert werden kann. Das Geld wird vom Mandanten nicht benötigt. Stichwort: Holding. Bitte zitiere die relevanten Paragraphen"
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
    - Generiere genau 1 Frage.
    - Wähle nur die Fragetypen aus, die für den gegebenen Kontext wirklich sinnvoll sind.
    - Formatiere die Ausgabe mit nur der Antwort nicht mehr und nicht weniger, keine Anderen Sätze oder ähnliches
    - Die Frage soll ausschließlich auf Deutsch geschrieben werden.
    - Nutze niemals die Beispielfragen, sondern lerne von ihnen.
    - Die Frage soll so ausführlich, wie möglich sein und wenn du Beispiele oder Szenarien nennst, sollten diese überaus detailiert sein.
    - Die Frage soll auf kreativen Weg formuliert werden, um den Kontext zu verstehen und die Fragen zu vermeiden, die bereits in der Antwort enthalten sind.
        """

        
        user_prompt = f"""
        Context: {article}

        Aufgabe:
        Erstelle eine detaillierte und präzise Frage, die sich auf das deutsche Steuerrecht bezieht. Wähle aus den Fragetypen den passendsten zu dem Kontext aus und erstelle eine Frage, welche mit dem Kontext beantwortbar ist. Sei Kreativ bei der Frageneinleitung und dem Szenario (sofern du eines erstellst). 

        Die Frage sollte:
        1. Mit dem Context beantwortbar sein
        2. Ins Detail des Contexts gehen
        3. Einen kreativen Einstieg/Anfang haben
        4. Sehr ausführlich, detailliert und kreativ sein
        5. So gestellt sein, wie ein Steuerberater sie auf der Arbeit an eine KI stellen würde, wenn er Hilfe bräuchte
        6. Da du die Frage wie ein Steuerberater stellst kannst du auch für deine Mandanten fragen

        Format:
        - WICHTIG: Gib nur die Frage aus, ohne zusätzliche Antworten oder Erklärungen.

        WICHTIG:
        - Antworte NIEMALS mit einer der Beispielfragen, wenn du findest dass du aus dem Kontext keine Frage generieren kannst antworte mit "Es tut mir Leid, aber der Kontext ist nicht ausreichend"
        - Die Frage MUSS mit dem Kontext beantwortbar sein und keine weiteren Informationen benötigen

        Sprache:
        - Die Frage muss vollständig auf Deutsch formuliert sein.
        - Achte darauf, dass die Frage klar und präzise ist, um die Antwort in eine juristische Richtung zu lenken.
        """
        temperature, top_p = self.rand_temp_tp()
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "model": self.model
        }
    
    def prepare_payload_a(self, article: str, frage: str) -> Dict:
        """Prepare the request payload for the FAU endpoint."""
        messages = [
            {"role": "system", "content": "Du bist ein hilfreicher Assistent für juristische Fragen. Beantworte die Frage basierend auf dem gegebenen Kontext. Wenn du dir bei der Antwort nicht sicher bist, gib dies an. Vermeide Aussagen, wie 'basierend auf gegebenem Kontext' oder 'aus dem gegebenem Text'. Wichtig ist, dass du sofern möglich wichtige Paragraphen, Rechtstexte oder Urteile zitierst. Beantworte daher alle Fragen präzise und stets korrekt. Es kommt auf Feinheiten an. Gib auch an, wie du auf die Lösung kommst. Der Mandant ist in einer schwierigen Situation, sei darum stets richtig in deinen Antworten und überprüfe die Antworten."},
            {"role": "user", "content": f"Kontext:\n{article}\n\nFrage: {frage}\n\n\n\n. Wenn die Antwort nicht aus dem Kontext hervorgeht, antworte nur mit: 'Es tut mir leid, aber der Kontext ist nicht ausreichend.'. Zitiere wenn möglich Rechtstexte, Paragraphen, Gerichtsurteile oder andere staatliche offiziele Quellen. Nutze nur Quellen die sich explizit auf Deutschland beziehen, nutze z.B. keine Quellen aus Österreich oder der Schweiz. Vermeide Sätze wie 'Es ist auch ratsam, sich von einem Steuerberater beraten zu lassen, um sicherzustellen, dass alle relevanten steuerlichen Vorschriften und Möglichkeiten berücksichtigt werden.' oder 'sollte idealerweise von einem Steuerberater oder einem Fachanwalt für Erbrecht durchgeführt werden.'."}
        ]
        temperature, top_p = self.rand_temp_tp()
        return {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "model": self.model
        }
    def rand_temp_tp(self):
        return random.uniform(0.3, 0.7), random.uniform(0.8, 0.9)
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()

    processor = Processor(args.config)
    processor.process()
