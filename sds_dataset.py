import os
import time
import pandas as pd
from tqdm import tqdm
import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Dict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class GeneratedQuestions(BaseModel):
    data: Dict[str, str]


def extract_doc_context(metadata: dict) -> str:
    # Delete below key-value pair from metadata: "_node_content", "_node_type", "document_id", "doc_id", "ref_doc_id"
    keys_to_remove = ["_node_content", "_node_type",
                      "document_id", "doc_id", "ref_doc_id"]
    for key in keys_to_remove:
        metadata.pop(key, None)

    hierarchy = metadata.get("categories", [])
    if metadata.get("title") is not None:
        hierarchy.append(metadata["title"])
    if metadata.get("section_title") is not None:
        hierarchy.append(metadata["section_title"])
    return f"{" > ".join(hierarchy)}: {metadata.get('description', '')}"


def create_db_engine():
    try:
        return create_engine(
            f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
            f"{os.getenv('DB_HOST')}:{os.getenv(
                'DB_PORT')}/{os.getenv('DB_NAME')}"
        )
    except SQLAlchemyError as e:
        logging.error(f"Error creating database engine: {e}")
        raise


def load_data_from_db(engine):
    query = f"SELECT text, metadata_ FROM {os.getenv('VECTOR_STORE_TABLE_NAME')}"
    try:
        return pd.read_sql_query(query, engine)
    except SQLAlchemyError as e:
        logging.error(f"Error executing query: {e}")
        raise


def generate_questions(client, prompt):
    try:
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct-AWQ",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "repetition_penalty": 1.05,
                "guided_json": GeneratedQuestions.model_json_schema(),
            },
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating question: {e}")
        return None


def main():
    engine = create_db_engine()
    df = load_data_from_db(engine)

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://10.0.0.25:8000/v1",
    )

    all_rows = []
    start_time = time.time()

    # Use tqdm to track progress across all rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Qestions"):
        doc_content = row["text"]
        doc_metadata = row["metadata_"]
        context = extract_doc_context(doc_metadata)

        prompt = f"""You are a helpful AI assistant. Your task is to generate questions from a human UI/UX designer's perspective based on the given content and its context. All content comes from the Adobe Spectrum Design Documentation.

## Input

- Context:
{context}
- Given Content:
{doc_content}

## Instructions

1. **Analyze Content**: Carefully review the key topics, facts, and concepts described in the given content. Focus only on the specific topic addressed in this content, regardless of the broader context.
2. **Generate Scope-Limited Questions**: Formulate questions a UI/UX designer might ask, strictly about the topic described in the given content.
   - Do Not generate questions solely based on the broader context or unrelated details.
   - Ensure all questions are tied directly to the information and concepts presented in the given content.
3. **Include a Mix Style of Questions**: Create a range of queries that cover both fundamental and advanced design considerations a UI/UX designer might have, such as understanding concepts, applying principles, and troubleshooting challenges.

## Output Format

Return the questions as a JSON object with the following structure:
```json
{{
  "1": "Generated question text",
  "2": "Generated question text",
  ...
}}
```
"""

        response_text = generate_questions(client, prompt)
        if response_text:
            try:
                questions = GeneratedQuestions.model_validate_json(response_text)
                for q in questions.data.values():
                    all_rows.append({
                        "anchor": q,
                        "positive": doc_content,
                        "metadata": doc_metadata
                    })
            except (ValidationError, ValueError) as e:
                logging.error(f"Error parsing response for row:\n{response_text}\n{e}")

    final_df = pd.DataFrame(all_rows, columns=["anchor", "positive", "metadata"])
    final_df.to_csv("synthetic_data.csv", index=False)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
