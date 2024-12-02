# D.AI (Data + AI)

D.AI is an open-source Streamlit application for both manual and AI-powered data analysis. Developed by **Tripathi Aditya Prakash**, it offers a seamless blend of traditional data exploration techniques and state-of-the-art AI-driven tools.

## Features

### 1. Manual Analysis
- Clean and preprocess data with options like removing duplicates, handling missing values, and transforming data.
- Perform statistical operations and generate correlation matrices.
- Visualize data interactively using Plotly and Seaborn.

### 2. AI-Powered Analysis
- Generate detailed insights and summaries using OpenAI's GPT.
- Create custom graphs with natural language prompts using the `seabornai` library.
- Analyze Python code in datasets with `Checkify`.
- Process and clean text data with the `cleantxty` library.

### 3. File Format Support
- **CSV**: Upload and explore spreadsheets easily.
- **Excel**: Handle `.xls` and `.xlsx` files.
- **JSON**: Parse and analyze structured data.
- **TXT**: Clean raw text data for NLP tasks.

### 4. Export Options
- Download processed datasets in Excel format.


## Installation

To run D.AI locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dai.git
   cd dai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run dai_app.py
   ```

4. Open your browser at `http://localhost:8501`.
