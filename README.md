# Spectrum Design Documentation Synthetic Dataset Generator

This repository leverages a local Large Language Model (LLM) to generate synthetic datasets for fine-tuning other LLMs. It extracts data from a vector store database, generates question-answer pairs based on provided prompts, and publishes the resulting dataset. You can find the published dataset 🤗[JianLiao/spectrum-design-docs](https://huggingface.co/datasets/spectrum_design_synthetic).

## Project Structure

```
├── publish_dataset.py     # Script to publish the dataset to Hugging Face Hub
├── requirements.txt       # List of dependencies
└── sds_dataset.py     # Main script to generate synthetic Q&A dataset
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
Use `pip` to install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory with the following variables:
```dotenv
DB_USER=<your-database-username>
DB_PASSWORD=<your-database-password>
DB_HOST=<your-database-host>
DB_PORT=<your-database-port>
DB_NAME=<your-database-name>
VECTOR_STORE_TABLE_NAME=<your-vector-store-table-name>
```

### 4. Configure OpenAI Client
Ensure the OpenAI client is set up with the correct API key and base URL in the `sds_dataset.py` file:
```python
client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.0.0.25:8000/v1",
)
```

## Running the Project

### Step 1: Generate Synthetic Dataset
Run the main script to generate question-answer pairs based on database content:
```bash
python sds_dataset.py
```
The output will be saved as `synthesis_qa_output.csv` in the root directory.

### Step 2: Publish Dataset to Hugging Face Hub
Use the `publish_dataset.py` script to push the generated dataset to your Hugging Face Hub repository:
```bash
python publish_dataset.py
```

## Key Notes

- **Database Access**: Ensure the database connection URL in `.env` is correctly configured.
- **OpenAI Client**: The OpenAI client uses a locally hosted LLM; ensure the API endpoint is reachable.
- **Customization**: Adjust prompts in the `sds_dataset.py` file to tailor Q&A generation to specific use cases.
- **Error Logging**: Errors during dataset generation are logged for debugging.

## Example Use Case
This repository is designed for teams looking to augment their fine-tuning datasets with synthetic data, particularly for UI design systems or documentation-heavy contexts.

## Dependencies
See `requirements.txt` for a list of dependencies, including:
- Hugging Face's `datasets` and `huggingface_hub`
- `pandas` for data manipulation
- `openai` for local LLM interaction
- `sqlalchemy` for database operations
- `python-dotenv` for environment variable management

## License
[MIT License](LICENSE)

