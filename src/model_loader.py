# src/model_loader.py (Updated for the new SOTA model)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import logging
import os

# --- Configuration ---
LOCAL_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_cache'))

def load_embedding_model(model_name='multi-qa-mpnet-base-dot-v1'):
    """
    Loads the new, high-performance sentence transformer model STRICTLY 
    from a local directory.
    """
    model_path = os.path.join(LOCAL_MODELS_DIR, model_name) 
    logging.info(f"Attempting to load local embedding model from: {model_path}")

    if not os.path.isdir(model_path):
        logging.error(f"FATAL: Model directory not found at '{model_path}'.")
        logging.error(f"Please make sure you have downloaded the model files and placed them in the '{model_name}' folder inside 'model_cache'.")
        return None
    
    try:
        # This model is packaged as a SentenceTransformer, so we can load it directly.
        model = SentenceTransformer(model_path)
        logging.info("High-performance local embedding model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load local embedding model from '{model_path}'. Error: {e}")
        return None

def load_generative_model():
    """
    Loads a generative language model (like Phi-2) from a local directory.
    (This function remains unchanged)
    """
    model_path = os.path.join(LOCAL_MODELS_DIR, 'phi-2')
    logging.info(f"Attempting to load local generative model from: {model_path}")

    if not os.path.isdir(model_path):
        logging.error(f"FATAL: Generative model directory not found at '{model_path}'.")
        return None, None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map=device, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logging.info("Local generative model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load local generative model. Error: {e}")
        return None, None