# Recommendation System

This recommendation system is an innovative project designed to enhance productivity and organization by leveraging artificial intelligence to manage and categorize digital content efficiently.

## Personalized Recommendations
This section generates recommendations based on the user's feedback—specifically, pins they have liked or disliked.

### How it works:

1. **User Feedback Collection**
   - As you interact with the app, your liked and disliked pins are stored in `st.session_state`.

2. **Embedding Generation**
   - Each pin's text content is embedded using the `all-MiniLM-L6-v2` model from `SentenceTransformer`.
   - These embeddings are normalized to prepare them for similarity comparison.

3. **Weighted User Profile**
   - Likes are given a weight of `+1`, while dislikes are weighted `-0.5` (configurable).
   - The system averages the embeddings from liked and disliked content.
   - A weighted profile vector is calculated as:  
     `weighted_profile = avg_liked - 0.5 * avg_disliked`
   - The profile vector is then normalized to maintain consistency in cosine similarity space.

4. **Semantic Search via FAISS**
   - The weighted profile is used to query a FAISS index of item embeddings.
   - The 5 most similar items are retrieved based on cosine similarity.
   - Results are displayed with their title and description for easy review.

This enables real-time, evolving recommendations that reflect what the user finds interesting—or not.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/EthanPeq/recommendation-system.git
    ```
2. Navigate to the project directory:
    ```bash
    cd recommendation_system
    ```
3. Create Virtual Environment
    ```
    py -3.11 -m venv venv
    ```
4. Activate Virtual Envirnoment
    ```
    .\venv\Scripts\Activate.ps1
    ```
5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:
    ```bash
    streamlit run src/recommendation_app.py
    ```
2. Follow the on-screen instructions to start organizing your content.


## Contact

For questions or feedback, please reach out to [epequignot.work@gmail.com](mailto:epequignot.work@gmail.com).
