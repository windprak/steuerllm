import socket
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoTokenizer
import uvicorn
import argparse

app = FastAPI()

# Initialize tokenizer
MODEL_NAME = "TechxGenus/Mistral-Large-Instruct-2407-AWQ"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TokenizeRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = None
    truncation: bool = False

class TokenCountRequest(BaseModel):
    text: str

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) != 0

def parse_port():
    parser = argparse.ArgumentParser(description="Start the tokenization server.")
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port number on which the server is to run (default: 8001)"
    )
    args = parser.parse_args()

    
    port = args.port
    
    # Check valid port range (1-65535)
    if not (1 <= port <= 65535):
        print(f"Ungültiger Port {port} angegeben. Benutze Standardport 8001.")
        return 8001

    if not is_port_available(port):
        print(f"Port {port} ist bereits belegt. Benutze Standardport 8001.")
        return 8001

    return port

@app.post("/tokenize")
async def get_tokenized(request: TokenizeRequest) -> Dict:
    try:
        # Ensure max_length is set when truncation is True
        if request.truncation and not request.max_length:
            request.max_length = 512  # Default max length
            
        # Process each text separately to maintain better control
        processed_texts = []
        for text in request.texts:
            # Tokenize
            tokens = tokenizer(
                text,
                max_length=request.max_length,
                padding=False,
                truncation=request.truncation,
                return_tensors="pt"
            )
            
            # Decode back to text
            decoded_text = tokenizer.decode(
                tokens['input_ids'][0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            processed_texts.append(decoded_text)
            
        return {"texts": processed_texts}
    except Exception as e:
        print(f"Error during tokenization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/count_tokens")
async def count_tokens(request: TokenCountRequest) -> Dict:
    try:
        # Tokenize the input text
        tokens = tokenizer(
            request.text,
            truncation=False,
            return_tensors="pt"
        )
        return {"count": tokens['input_ids'].shape[1]}  # Get token count for the sequence
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
