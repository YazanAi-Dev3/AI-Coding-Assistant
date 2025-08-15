# AI Coding Assistant with a Hybrid RAG Architecture

This project is a sophisticated, hybrid chatbot designed to function as an intelligent programming assistant. It leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide answers to coding questions with high accuracy and efficiency.

The system is designed to run entirely offline, utilizing locally stored models, and features a self-improving knowledge base.

##  Key Features

-   **Hybrid RAG System:** Intelligently switches between retrieving pre-existing answers for known topics and generating new answers for novel questions, ensuring both speed and accuracy.
-   **High-Accuracy Retrieval:** Uses a state-of-the-art `multi-qa-mpnet-base-dot-v1` model, fine-tuned for semantic search, to find the most relevant answers from its knowledge base with high precision.
-   **On-Demand Answer Generation:** Leverages the powerful `microsoft/phi-2` generative model to create new, coherent answers for questions not found in the database.
-   **Self-Improving Knowledge Base:** The system is designed with a mechanism to learn from new questions. When a new answer is generated, the corresponding question and answer pair can be added back to the database, making the system smarter over time.
-   **Fully Offline Inference:** After the initial model download, the entire system runs locally without needing an active internet connection.
-   **Modular Codebase:** The project is structured with clean, refactored Python scripts, separating concerns for model loading, database construction, and chatbot logic.

---

##  System Architecture

The chatbot operates on a two-path logic:

1.  A user's question is first converted into a vector embedding using the retriever model.
2.  This vector is compared against a pre-computed vector database of known questions.
3.  **If a match with a high similarity score is found:** The corresponding high-quality, pre-written answer is instantly retrieved and returned.
4.  **If no confident match is found:** The question is passed to the generative model (Phi-2), which generates a new answer from scratch. This new question-answer pair can then be integrated back into the vector database.

---

##  Models Used

This project relies on two main models that must be downloaded manually.

-   **Embedding Model (Retriever):** `sentence-transformers/multi-qa-mpnet-base-dot-v1`
-   **Generative Model (Generator):** `microsoft/phi-2`

**Note:** Due to their large size, the model files are **not** included in this repository. Please follow the setup instructions below to download and place them correctly.

---

##  Setup and Installation

To run this project on your local machine, please follow these steps carefully:

1.  **Clone the Repository:**
    ```sh
    git clone [https://github.com/YazanAi-Dev3]
    cd AI-Coding-Assistant
    ```

2.  **Download the Models Manually:**
    -   Go to the Hugging Face Hub and download the files for both models listed above.
    -   Create a folder named `model_cache` in the main project directory.
    -   Inside `model_cache`, create two subfolders: `multi-qa-mpnet-base-dot-v1` and `phi-2`.
    -   Place the corresponding downloaded files for each model into its respective folder.

3.  **Create a Virtual Environment & Install Dependencies:**
    ```sh
    # Create and activate a virtual environment
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate

    # Install the required libraries
    pip install -r requirements.txt
    ```

4.  **Build the Vector Database:**
    -   Ensure your initial `questions_answers.json` file is in the `data/` folder.
    -   Run the build script from the main project directory:
    ```sh
    python src/build_vector_db.py
    ```
    This will create the `vector_database.pkl` file using your local CodeBERT/MPNet model.

---

##  How to Use

Once the setup is complete, you can run the interactive demo:

-   Open and run the cells in the **`Demo.ipynb`** notebook. This will load the chatbot and test it on sample questions, showcasing both the retrieval and generation functionalities.
