#!/usr/bin/env python3
"""
Mock LLM Server - Simulates OpenAI-compatible API for testing.
Returns realistic conversation JSON without requiring actual LLM.
"""
import json
import time
import random
from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample conversation templates
CONVERSATION_TEMPLATES = [
    [
        {"role": "user", "content": "Können Sie mir die wichtigsten Punkte zu {topic} erklären?"},
        {"role": "assistant", "content": "Gerne erkläre ich Ihnen die wesentlichen Aspekte zu {topic}. Die Hauptpunkte sind: 1) {point1}, 2) {point2}, und 3) {point3}. Diese Regelungen sind besonders wichtig für die praktische Anwendung."}
    ],
    [
        {"role": "user", "content": "Was bedeutet die Regelung zu {topic} konkret?"},
        {"role": "assistant", "content": "Die Regelung zu {topic} besagt im Wesentlichen, dass {explanation}. Dies hat direkte Auswirkungen auf {impact}."},
        {"role": "user", "content": "Gibt es dabei Ausnahmen zu beachten?"},
        {"role": "assistant", "content": "Ja, es gibt einige wichtige Ausnahmen: {exceptions}. In der Praxis sollten Sie besonders auf {consideration} achten."}
    ],
    [
        {"role": "user", "content": "Welche steuerlichen Konsequenzen hat {topic}?"},
        {"role": "assistant", "content": "Bei {topic} ergeben sich folgende steuerliche Konsequenzen: {consequences}. Es ist wichtig, diese Aspekte bei der Planung zu berücksichtigen."},
        {"role": "user", "content": "Wie wirkt sich das auf Unternehmen aus?"},
        {"role": "assistant", "content": "Für Unternehmen bedeutet dies konkret: {business_impact}. Besonders zu beachten sind dabei die Fristen und Meldepflichten."}
    ]
]


def generate_conversation(volltext: str, titel: str, themengebiet: str) -> list:
    """Generate a realistic conversation based on input."""
    template = random.choice(CONVERSATION_TEMPLATES)
    
    topic = titel if titel else "das Thema"
    area = themengebiet if themengebiet else "Steuerrecht"
    
    text_snippet = volltext[:200] if volltext else "relevante Regelungen"
    
    conversation = []
    for msg in template:
        content = msg["content"].format(
            topic=topic,
            point1=f"gesetzliche Grundlagen aus dem {area}",
            point2="praktische Anwendungsfälle",
            point3="mögliche Ausnahmen und Sonderfälle",
            explanation=f"gemäß der Vorschriften im {area}",
            impact="die steuerliche Behandlung",
            exceptions="spezielle Konstellationen bei bestimmten Sachverhalten",
            consideration="die Dokumentationspflichten",
            consequences="sowohl positive als auch negative Auswirkungen",
            business_impact="eine Anpassung der internen Prozesse"
        )
        conversation.append({"role": msg["role"], "content": content})
    
    return conversation


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Mock OpenAI chat completions endpoint."""
    data = request.get_json()
    
    time.sleep(random.uniform(0.1, 0.5))
    
    messages = data.get('messages', [])
    user_message = next((m for m in messages if m['role'] == 'user'), None)
    
    if not user_message:
        return jsonify({
            "error": {
                "message": "No user message found",
                "type": "invalid_request_error"
            }
        }), 400
    
    user_content = user_message.get('content', '')
    
    volltext = ""
    titel = ""
    themengebiet = ""
    
    for line in user_content.split('\n'):
        if 'Volltext:' in line:
            volltext = line.split('Volltext:')[-1].strip()
        elif 'Titel:' in line:
            titel = line.split('Titel:')[-1].strip()
        elif 'Themengebiet:' in line:
            themengebiet = line.split('Themengebiet:')[-1].strip()
    
    conversation = generate_conversation(volltext, titel, themengebiet)
    
    response_text = json.dumps(conversation, ensure_ascii=False)
    
    response = {
        "id": f"chatcmpl-mock-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get('model', 'mock-model'),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300
        }
    }
    
    return jsonify(response)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "mock-llm-server"})


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with info."""
    return jsonify({
        "service": "Mock LLM Server",
        "version": "1.0",
        "endpoints": {
            "/v1/chat/completions": "POST - Chat completions (OpenAI compatible)",
            "/health": "GET - Health check"
        }
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Mock LLM Server Starting...")
    print("=" * 60)
    print("Listening on: http://localhost:6000")
    print("API Endpoint: http://localhost:6000/v1/chat/completions")
    print("Health Check: http://localhost:6000/health")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=6000, debug=False)
