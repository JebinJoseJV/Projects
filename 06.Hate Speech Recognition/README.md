This directory contains the code and resources for a hate speech recognition application built using Streamlit.

## Project Structure

*   **`app.py`**: The main Streamlit application file.  Run this to launch the web app.
*   **`hate_speech_recognition.ipynb`**: Jupyter Notebook containing exploratory data analysis (EDA) and model training procedures.
*   **`requirements.txt`**:  List of project dependencies.  Use `pip install -r requirements.txt` to set up your environment.


## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

This will launch the application in your web browser.


## Usage

The application allows users to input text and get a prediction of whether the text contains hate speech.

