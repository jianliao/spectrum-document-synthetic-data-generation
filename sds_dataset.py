import os
import time
import re
import pandas as pd
from tqdm import tqdm
import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class QandA(BaseModel):
    question: str
    answer: str

class QandAs(BaseModel):
    data: list[QandA]

# Database connection URL
try:
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
except SQLAlchemyError as e:
    logging.error(f"Error creating database engine: {e}")
    raise

# Query to extract data
query = f"SELECT text, metadata_ FROM {os.getenv('VECTOR_STORE_TABLE_NAME')}"

# Load data into a pandas DataFrame
try:
    df = pd.read_sql_query(query, engine)
except SQLAlchemyError as e:
    logging.error(f"Error executing query: {e}")
    raise

# Initialize OpenAI client
client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.0.0.25:8000/v1",
)

all_rows = []

# Record start time
start_time = time.time()

# Use tqdm to track progress across all rows
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Q&A"):
    doc_content = row["text"]
    doc_metadata = row["metadata_"]

    prompt = f"""Create questions that a UI designer might ask based on provided content from the Spectrum design system documentation, and generate corresponding answers. Present each question-answer pair as a JSON object within a JSON array.

# Steps
1. **Review the Content**: Carefully read and analyze the provided Spectrum design system documentation.
2. **Identify Key Information**: Identify important themes, guidelines, and potentially ambiguous or complex areas within the documentation.
3. **Formulate Questions**: Develop questions that a UI designer might naturally have when reading or applying the documentation.
4. **Answer Questions**: Based on your analysis of the content, generate suitable answers for each question.

# Output Format
The output should be a JSON array where each element is an object containing "question" and "answer" keys.

# Notes

- Ensure that each question clearly relates to the provided documentation content.
- The answers should directly address the questions in a concise manner.
- Consider including a mix of common queries and more in-depth questions to cover different levels of user inquiry.

# Spectrum Documentation
{doc_content}
"""

    # Generate Q&A
    try:
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct-AWQ",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant with "
                        "deep expertise in UI system design, familiar with foundational concepts such "
                        "as components, color, and typography."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "repetition_penalty": 1.05,
                "guided_json": QandAs.model_json_schema(),
            },
        )
    except Exception as e:
        logging.error(f"Error generating Q&A: {e}")
        continue

    # Extract the raw response
    response_text = chat_response.choices[0].message.content.strip()

    # Parse JSON into our data model
    try:
        qandas = QandAs.model_validate_json(response_text)
        for qa in qandas.data:
            all_rows.append({
                "question": qa.question,
                "answer": qa.answer,
                "text": doc_content,
                "metadata": doc_metadata
            })
    except (ValidationError, ValueError) as e:
        logging.error(f"Error parsing response for row:\n{response_text}\n{e}")

# Create a new DataFrame with the collected rows
final_df = pd.DataFrame(all_rows, columns=["question", "answer", "text", "metadata"])
final_df.to_csv("synthesis_qa_output.csv", index=False)

# Calculate and print total time
end_time = time.time()
total_time = end_time - start_time
logging.info(f"Total time: {total_time:.2f} seconds")
