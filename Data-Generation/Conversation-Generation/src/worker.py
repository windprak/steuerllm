#!/usr/bin/env python3
"""
Worker script for generating conversational datasets from JSON chunks.
Processes input JSON files and generates conversations using an LLM API.
"""
import argparse
import asyncio
import sys
import json
import logging
import os
from typing import List, Dict
import yaml

import aiohttp
from aiofiles import open as aio_open
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_sys_msg():
    """Return the system message for the LLM."""
    return """Du bist ein hochspezialisierter virtueller Steuerexperte, der sich darauf spezialisiert hat, steuerrechtliche Themen klar, präzise und für Laien verständlich zu erklären. Deine Aufgabe ist es, realistische, thematisch kohärente und nachvollziehbare Konversationen zu erstellen. 

Folgende Richtlinien musst du einhalten:
1. **Zielgruppe**: Die Nutzer (user) stellen Fragen, die sowohl von Laien als auch von steuerrechtlich versierten Personen stammen können. Deine Antworten sollten immer professionell und fachlich korrekt sein, aber auch leicht verständlich bleiben.
2. **Konversation**: Erstelle ein realistisches Gespräch basierend auf vorgegebenen Themen und Inhalten. Die Länge der Konversation richtet sich nach der Komplexität des Themas und kann flexibel sein.
3. **Formatierung**: Die Konversation wird in einem JSON-Format dargestellt, mit klaren Rollen (user und assistant) und verständlichen Inhalten. Nutze ein iteratives Frage-Antwort-Muster.
4. **Zusammenhänge**: Gib den Nutzern genügend Kontext, damit sie die Konversation nachvollziehen können, auch ohne direkten Zugriff auf die Originalquelle (den Textchunk).
5. **Chain of Thought**: Generiere immer eine "rote Linie", um die Hauptpunkte und den logischen Fluss des Gesprächs vorab zu skizzieren. Nutze diese als Grundlage für den Dialog.
6. **Stil der Fragen**: Benutze keine Floskeln wie "Ich habe eine Frage" oder "Ich habe gehört, dass". Stelle sicher, dass die Fragen abwechslungsreich und natürlich sind.

Bleibe sachlich und thematisch genau, aber auch flexibel, um auf spezifische Szenarien einzugehen.
"""


def get_usr_msg(text: str, themengebiet: str, topic_filter: str, titel: str, autor: str, volltext: str):
    """Generate the user message for the LLM."""
    template = """
Du bist ein Experte für Steuerberatung und sollst eine Konversation basierend auf steuerrechtlichen Themen generieren. Dir werden die folgenden Informationen zu einem Textchunk gegeben:

- Themengebiet: {themengebiet}
- TopicFilter: {topic_filter}
- Titel: {titel}
- Autor: {autor}
- Volltext: {volltext}

Erstelle basierend auf diesen Informationen:
1. **Rote Linie (Chain of Thought)**: Eine kurze Beschreibung des Gesprächsverlaufs. Diese sollte die Hauptpunkte des Dialogs zusammenfassen und den logischen Fluss des Gesprächs skizzieren. Sie dient als Plan für den Dialog.
2. **Konversation**: Eine realistische und thematisch kohärente Unterhaltung zwischen einem Nutzer (user), der steuerrechtliche Fragen stellt, und einer KI (assistant), die als Steuerberater fungiert. Die Länge der Konversation hängt von der Komplexität des Themas im Volltext ab und kann flexibel sein (mindestens 1, aber auch mehr als 2 Interaktionen).

**Wichtige Hinweise zur Konversation**:
- **Variation**: Stelle sicher, dass die Formulierungen der Nutzerfragen abwechslungsreich und natürlich sind. Vermeide häufige Einleitungen wie "Ich habe eine Frage" oder "Ich habe gehört, dass". Stattdessen soll der Nutzer auch präzisere oder spezifischere Fragen stellen (z. B. direkt auf ein Szenario eingehen oder eine Regelung hinterfragen).
- **Natürlichkeit**: Die Fragen und Antworten sollen realistische Dialoge widerspiegeln, wie sie in einer echten Steuerberatungssituation vorkommen könnten.
- **Flexibilität**: Die Länge der Konversation hängt von der Komplexität des Themas im Volltext ab und kann flexibel sein (mindestens 1, aber auch mehr als 2 Interaktionen).

Gib die Konversation im JSON-Format aus. Jede Interaktion soll durch ein Objekt dargestellt werden. 

**Format**:
json '''
[
  {{ "content": "string", "role": "user" }},
  {{ "content": "string", "role": "assistant" }},
  ...
]
'''

**Beispiel für eine rote Linie**:
"Der Nutzer stellt Fragen zu steuerrechtlichen Konsequenzen einer neuen Gesetzgebung im Bereich {themengebiet}. Die KI erläutert die Regelung, gibt Beispiele und erklärt mögliche Auswirkungen auf Unternehmen und Privatpersonen. Es folgen spezifische Fragen zur praktischen Umsetzung."

Erstelle zuerst eine "rote Linie" für den gegebenen Textchunk. Nutze diese, um die Konversation zu strukturieren, und generiere dann die vollständige Unterhaltung. Achte darauf, dass der Dialog nachvollziehbar bleibt, auch ohne direkten Zugriff auf den Volltext. Beachte die erwähnte Flexibitlät der Fragenstellung.
""".replace('%', '%%')

    return template.format(
        themengebiet=themengebiet,
        topic_filter=topic_filter,
        titel=titel,
        autor=autor,
        volltext=volltext
    )


class RequestQueue:
    """Manages a queue of requests with retry logic and concurrency control."""
    
    def __init__(self, max_concurrent: int, config: dict, client: AsyncOpenAI, output_file: str):
        self.max_concurrent = max_concurrent
        self.config = config
        self.client = client
        self.output_file = output_file
        self.pending_queue = asyncio.Queue()
        self.active_requests = set()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.successful_requests = 0
        self.failed_requests = 0

    async def add_request(self, data: Dict):
        """Add a request to the queue with retry logic."""
        record_id = data['_recordid']
        data['attempt'] += 1
        
        max_retries = self.config['api']['max_retries']
        if data['attempt'] <= max_retries:
            logger.warning(f"Retry for request {record_id}, attempt {data['attempt']}")
            await self.pending_queue.put(data)
            await self.process_queue()
        else:
            self.failed_requests += 1
            logger.error(f"Max retries reached for request {record_id}")

    async def process_queue(self):
        """Process queued requests up to max concurrent limit."""
        tasks = []
        while (not self.pending_queue.empty() and 
               len(self.active_requests) < self.max_concurrent):
            try:
                data = self.pending_queue.get_nowait()
                task = asyncio.create_task(self.process_request(data))
                self.active_requests.add(task)
                tasks.append(task)
                task.add_done_callback(self.active_requests.discard)
            except asyncio.QueueEmpty:
                break

        if tasks:
            await asyncio.gather(*tasks)

    async def process_request(self, data: Dict):
        """Process a single request with the LLM API."""
        try:
            async with self.semaphore:
                record_id = data.get('_recordid', 'unknown')
                
                if not data.get('volltext'):
                    logger.error(f"Request {record_id} missing required field: volltext")
                    self.failed_requests += 1
                    return

                try:
                    messages = [
                        {"role": "system", "content": get_sys_msg()},
                        {"role": "user", "content": get_usr_msg(
                            data.get('text', ''),
                            data.get('themengebiet', ''),
                            data.get('topic_filter', ''),
                            data.get('titel', ''),
                            data.get('autor', ''),
                            data['volltext']
                        )}
                    ]

                    request_body = {
                        "model": self.config['api']['model'],
                        "messages": messages,
                        "temperature": self.config['worker']['temperature'],
                        "top_p": self.config['worker']['top_p']
                    }

                    timeout = self.config['api']['timeout']
                    completion = await asyncio.wait_for(
                        self.client.chat.completions.create(**request_body),
                        timeout=timeout
                    )
                    
                    content = completion.choices[0].message.content.strip()
                    
                    try:
                        json_content = json.loads(content)
                    except json.JSONDecodeError:
                        if "```json" in content:
                            json_content = content.split("```json")[1].split("```")[0].strip()
                            json_content = json.loads(json_content)
                        else:
                            raise ValueError("Response does not contain valid JSON")
                    
                    processed_content = {
                        '_recordid': record_id,
                        'conversation': json_content
                    }
                    
                    async with aio_open(self.output_file, mode='a', encoding='utf-8') as fout:
                        await fout.write(json.dumps(processed_content, ensure_ascii=False) + '\n')
                    
                    logger.info(f"Request {record_id} completed successfully")
                    self.successful_requests += 1
                    return processed_content
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for request {record_id}")
                    if data['attempt'] < self.config['api']['max_retries']:
                        data['attempt'] += 1
                        await self.add_request(data)
                    else:
                        self.failed_requests += 1
                        
                except Exception as e:
                    logger.error(f"API call failed for request {record_id}: {str(e)}")
                    if data['attempt'] < self.config['api']['max_retries']:
                        data['attempt'] += 1
                        await self.add_request(data)
                    else:
                        self.failed_requests += 1

        except Exception as final_error:
            logger.critical(f"Unhandled error in process_request: {str(final_error)}")
            self.failed_requests += 1
            return None


async def load_processed_records(output_path: str) -> set:
    """Load already processed record IDs from output file."""
    processed_records = set()
    try:
        if os.path.exists(output_path):
            async with aio_open(output_path, 'r', encoding='utf-8') as file:
                async for line in file:
                    try:
                        record = json.loads(line)
                        if '_recordid' in record:
                            processed_records.add(record['_recordid'])
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Found {len(processed_records)} already processed records")
    except Exception as e:
        logger.error(f"Error loading processed records: {str(e)}")
    
    return processed_records


async def parse_file(file_path: str, request_queue: RequestQueue, processed_records: set):
    """Parse input JSON file and add items to request queue."""
    try:
        async with aio_open(file_path, 'r', encoding='utf-8') as file:
            content = await file.read()
            data = json.loads(content)
            
            for index, chunk in enumerate(data):
                try:
                    if chunk['_recordid'] in processed_records:
                        logger.debug(f"Skipping already processed record {chunk['_recordid']}")
                        continue

                    if 'Volltext' not in chunk or not chunk['Volltext']:
                        logger.error(f"Skipping chunk {index} - missing Volltext")
                        continue

                    request_data = {
                        '_recordid': chunk['_recordid'],
                        'text': chunk.get('text', ''),
                        'themengebiet': chunk.get('Themengebiet', ''),
                        'topic_filter': chunk.get('TopicFilter', ''),
                        'titel': chunk.get('Titel', ''),
                        'autor': chunk.get('Autor', ''),
                        'volltext': chunk['Volltext'],
                        'attempt': 0
                    }

                    await request_queue.add_request(request_data)
                except Exception as e:
                    logger.error(f"Error processing chunk {index}: {str(e)}")
                    continue
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")


async def main(args, config):
    """Main function to coordinate file parsing and conversation generation."""
    base_url = f"{config['api']['base_url']}/v1/"
    
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=config['api']['api_key'],
        timeout=config['api']['timeout']
    )

    logger.info(f"Base URL: {base_url}")
    logger.info(f"Max Concurrent Requests: {args.max_concurrent}")

    file_list = args.input_files.split(',')
    
    for file_path in file_list:
        file_path = file_path.strip()
        input_basename = os.path.basename(file_path)
        input_name = os.path.splitext(input_basename)[0]
        output_path = os.path.join(args.output_dir, f"{input_name}.jsonl")
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        processed_records = await load_processed_records(output_path) if config['data']['skip_processed'] else set()
        
        request_queue = RequestQueue(
            args.max_concurrent, 
            config, 
            client, 
            output_path
        )
        
        await parse_file(file_path, request_queue, processed_records)
        await request_queue.process_queue()
        
        total = request_queue.successful_requests + request_queue.failed_requests
        if total > 0:
            success_rate = (request_queue.successful_requests / total) * 100
            logger.info(f"Processing complete. Total: {total}, Success rate: {success_rate:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conversational datasets from JSON chunks")
    parser.add_argument('--input-files', required=True, help='Comma-separated list of input JSON files')
    parser.add_argument('--output-dir', required=True, help='Directory for output JSONL files')
    parser.add_argument('--config', default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--max-concurrent', type=int, help='Override max concurrent requests from config')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.max_concurrent:
        config['worker']['max_concurrent'] = args.max_concurrent
    
    if not args.max_concurrent:
        args.max_concurrent = config['worker']['max_concurrent']
    
    log_level = getattr(logging, config['logging']['level'])
    logging.getLogger().setLevel(log_level)
    
    asyncio.run(main(args, config))
