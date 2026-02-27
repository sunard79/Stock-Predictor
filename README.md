# Stock-Predictor

This project aims to predict stock prices using various data analysis and machine learning techniques. It leverages the Gemini 2.0 Flash model for certain generative tasks within the project workflow, such as data summarization or generating insights.

## Repository

The repository for this project is available at: [https://github.com/sunard79/Stock-Predictor](https://github.com/sunard79/Stock-Predictor)

## Project Structure

- `data/`: Contains raw and processed financial data.
  - `data/raw/`: Raw, untransformed data.
  - `data/processed/`: Cleaned and preprocessed data ready for analysis.
- `notebooks/`: Jupyter notebooks for exploratory data analysis, model development, and experimentation.
- `src/`: Python source code for data collection, preprocessing, model training, and prediction.
- `database/`: Stores SQLite databases for efficient data management.
- `config/`: Configuration files for the project.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sunard79/Stock-Predictor.git
    cd Stock-Predictor
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    ./venv/Scripts/activate  # On Windows
    source venv/bin/activate # On macOS/Linux
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Create a `.env` file in the root directory for any sensitive information or API keys.

## Data Collection

The `src/data_collection.py` script is responsible for downloading historical stock data and storing it in a local SQLite database.

## Usage

Further instructions on running specific scripts, notebooks, and models will be provided here as the project develops.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

[Specify your license here, e.g., MIT, Apache 2.0]
