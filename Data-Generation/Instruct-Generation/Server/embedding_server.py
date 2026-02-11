import argparse
import socket
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import uvicorn

app = FastAPI()

# Initialize model and tokenizer
MODEL_NAME = "intfloat/multilingual-e5-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

class TextRequest(BaseModel):
    texts: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Sample text 1", "Sample text 2"]
            }
        }

def average_pool(last_hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) != 0

def parse_port():
    parser = argparse.ArgumentParser(description="Start the embedding server.")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number on which the server is to run (default: 8001)"
    )
    args = parser.parse_args()

    
    port = args.port
    print(port)
    # Check valid port range (1-65535)
    if not (1 <= port <= 65535):
        print(f"Ungültiger Port {port} angegeben. Benutze Standardport 8001.")
        return 8000

    if not is_port_available(port):
        print(f"Port {port} ist bereits belegt. Benutze Standardport 8001.")
        return 8000

    return port

@app.post("/embed")
async def get_embeddings(request: TextRequest) -> Dict[str, List[List[float]]]:
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
            
        # Tokenize and encode texts
        batch_dict = tokenizer(
            request.texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        # Convert to numpy and then explicitly convert to float values
        embeddings_np = embeddings.cpu().numpy()
        embeddings_list = [[float(x) for x in emb] for emb in embeddings_np.tolist()]
        
        return {"embeddings": embeddings_list}
    except Exception as e:
        print(f"Error in get_embeddings: {str(e)}")  
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = parse_port()
    uvicorn.run(app, host="0.0.0.0", port=port)
