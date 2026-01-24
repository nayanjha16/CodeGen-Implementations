# SMART SLM Query Interface

## Overview

This is a capstone project that demonstrates the conversion of relational SQL data to NoSQL (MongoDB-style) format and provides a natural language query interface powered by Google's Generative AI. The project focuses on debit card transaction data, implementing a SMART (Schema-Aware MongoDB Query Translator) framework for generating NoSQL queries from natural language inputs.

The application consists of:
- **Data Migration Module**: Converts flat SQL JSON data to denormalized NoSQL collections
- **Query Interface**: A Streamlit web app that accepts natural language queries and returns MongoDB query results
- **AI-Powered Schema Prediction**: Uses Google Gemini AI to interpret user queries and generate appropriate MongoDB aggregation pipelines

## Features

- **Data Conversion**: Automated transformation from SQL schema to NoSQL collections with proper denormalization
- **Natural Language Processing**: Convert plain English queries to MongoDB queries using AI
- **Mock Database Support**: In-memory MongoDB-like database for demonstration and testing
- **File Upload Support**: Dynamic schema and database loading via file uploads
- **Streamlit Interface**: User-friendly web interface for query input and results display
- **Google GenAI Integration**: Leverages Gemini Pro for intelligent query generation

## Project Structure

```
smartslm/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── smart/
│   ├── slm.py                      # SMART framework implementation with GenAI
│   ├── smart.py                    # Core SMART logic
│   ├── utils.py                    # Utility functions
│   ├── feeback.py                  # Schema prediction feedback
│   ├── dataloading/
│   │   ├── mongodbmock.py          # Mock MongoDB implementation
│   │   ├── sqldb.py                # SQLite database setup
│   │   ├── sqlfinder.py            # SQL query finding utilities
│   │   └── SqlLightLoad.py         # SQLite data loading
│   └── dbcreation/
│       └── SQLtoNoSQLDbCreation.py # SQL to NoSQL conversion script
├── debit_card_specializing_sql_data.json    # Sample SQL data
├── Nosql_debit_card_specializing_data.json  # Converted NoSQL data
└── Various output JSON files for testing
```

## Installation

1. **Clone the repository** (if applicable) or ensure you have the project files.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google API Key**:
   - Obtain a Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set the environment variable:
     ```bash
     export GOOGLE_API_KEY=your_api_key_here
     ```
   - For Windows PowerShell:
     ```powershell
     $env:GOOGLE_API_KEY="your_api_key_here"
     ```

## Usage

### Running the Application

1. **Local Development**:
   ```bash
   streamlit run app.py
   ```

2. **Access the app** at `http://localhost:8501`

### Using the Query Interface

1. **Upload Schema**: Upload a JSON file containing the database schema
2. **Upload NoSQL Database**: Upload the converted NoSQL data in JSON format
3. **Optional: Upload SQLite DB**: For SQL database comparison
4. **Enter Natural Language Query**: Type your query in plain English (e.g., "Show me transactions for customer 123 in 2023")
5. **View Results**: The app will generate and execute the corresponding MongoDB query

### Data Migration

To convert SQL data to NoSQL format:

1. Prepare your SQL data in JSON format (see `debit_card_specializing_sql_data.json` for example)
2. Run the conversion script:
   ```python
   from smart.dbcreation.SQLtoNoSQLDbCreation import sql_to_nosql_converter
   sql_to_nosql_converter('input_sql_data.json', 'output_nosql_data.json')
   ```

The script performs:
- Customer denormalization with embedded consumption data
- Product reference collection creation
- Transaction linking
- Gas station data transformation

## Deployment

### Hugging Face Spaces

1. Create a new Hugging Face Space at [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose "Streamlit" as the SDK
3. Upload the following files:
   - `app.py`
   - `requirements.txt`
   - `smart/` folder (entire directory)
   - `nosql_final_full.json`
   - Sample data files as needed
4. In the Space settings, add a secret named `GOOGLE_API_KEY` with your Google GenAI API key
5. The app will deploy automatically

### Other Platforms

The app can be deployed on any platform supporting Streamlit:
- Heroku
- AWS
- Google Cloud
- Azure

## Dependencies

- `streamlit`: Web app framework
- `google-genai`: Google Generative AI client
- `pymongo`: MongoDB driver (for mock implementation)
- `numpy`, `scikit-learn`: Data processing
- `sentence-transformers`: Text embeddings
- `tqdm`: Progress bars

## Security Notes

- Never commit API keys to version control
- Use environment variables or secure secret management for API keys
- The fallback API key in `slm.py` is for demonstration only - replace with your own

## Contributing

This is a capstone project. For improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Specify your license here, e.g., MIT License]

## Acknowledgments

- Google Generative AI for powering the natural language processing
- Streamlit for the web interface
- Capstone project requirements and guidance