# src/build_vector_db.py (Final Simplified Version)

import json
import joblib
import logging
from model_loader import load_embedding_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(message)s')

INPUT_JSON_PATH = 'data/questions_answers.json'
OUTPUT_DB_PATH = 'database/vector_database.pkl'

def build_and_save_database():
    logging.info("Starting the database build process with the new model...")

    # 1. Load the ready-to-use embedding model
    embedding_model = load_embedding_model()
    if embedding_model is None:
        logging.error("Could not proceed without the embedding model. Exiting.")
        return

    # 2. Load the raw Q&A data
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    logging.info(f"Successfully loaded {len(qa_data)} Q&A pairs.")

    # 3. Encode questions using the model's simple .encode() method
    questions = [item['question'] for item in qa_data]
    answers = [item['answer'] for item in qa_data]

    logging.info("Encoding all questions... This may take some time.")
    # The .encode() method handles tokenization, pooling, etc., all in one step.
    question_embeddings = embedding_model.encode(questions, show_progress_bar=True)
    logging.info("All questions have been encoded successfully.")

    vector_database = {
        'questions': questions,
        'answers': answers,
        'embeddings': question_embeddings
    }

    # 4. Save the final database
    with open(OUTPUT_DB_PATH, 'wb') as f:
        joblib.dump(vector_database, f)
    logging.info(f"Vector database successfully created and saved to '{OUTPUT_DB_PATH}'")

if __name__ == "__main__":
    build_and_save_database()