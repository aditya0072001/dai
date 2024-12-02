import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from checkify.code_checker import check_code
from cleantxty import clean
import seaborn as sns
import matplotlib.pyplot as plt
import openai


# Set OpenAI API Key
def set_openai_api_key(api_key):
    openai.api_key = api_key


# Generate Graph Function
def generate_graph(data, prompt, style="whitegrid", size=(6, 4)):
    if not openai.api_key:
        raise ValueError("OpenAI API key not set. Please use `set_openai_api_key` function to set the API key.")

    # Generate text response from OpenAI
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    generated_text = response.choices[0].text.strip()

    # Ensure dataframe has sufficient columns for plotting
    if len(data.columns) < 2:
        raise ValueError("Dataset must have at least two columns for plotting.")

    # Plotting logic
    sns.set(style=style)
    sns.set(rc={"figure.figsize": size})
    sns.barplot(data=data, x=data.columns[0], y=data.columns[1])
    return plt.gcf()  # Return the current figure for rendering in Streamlit


# Streamlit App
API_KEY = st.text_input("Enter your OpenAI API Key:", type="password")
if API_KEY:
    set_openai_api_key(API_KEY)

st.title("D.AI (Data + AI)")
st.write("Analyze your datasets with ease using **D.AI**. Choose between traditional manual methods and cutting-edge AI-powered insights.")
uploaded_files = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON, or TXT)", accept_multiple_files=True)

analysis_type = st.radio("Select Your Analysis Mode:", [
                         "Manual Analysis", "AI-Powered Analysis"])
st.title("D.AI (Data + AI)")
st.write("Analyze your datasets with ease using **D.AI**. Choose between traditional manual methods and cutting-edge AI-powered insights.")
uploaded_files = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON, or TXT)", accept_multiple_files=True)

analysis_type = st.radio("Select Your Analysis Mode:", [
                         "Manual Analysis", "AI-Powered Analysis"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        st.write(f"Processing file: {uploaded_file.name}")

        # Load the file based on extension
        try:
            if file_extension == 'csv':
                dataframe = pd.read_csv(uploaded_file)
            elif file_extension in ['xls', 'xlsx']:
                dataframe = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                dataframe = pd.read_json(uploaded_file)
            elif file_extension == 'txt':
                with open(uploaded_file, 'r') as f:
                    text_data = f.read()
                    st.write("Text file detected. Clean the text using cleantxty:")
                    st.text_area("Raw Text", text_data)
                    cleaned_text = clean(text_data)
                    st.text_area("Cleaned Text", cleaned_text)
                continue
            else:
                st.error("Unsupported file format!")
                continue
        except Exception as e:
            st.error(f"Error loading file: {e}")
            continue

        # Display file stats and preview
        st.write(f"**Shape:** {dataframe.shape}")
        st.write("**Data Preview:**")
        st.dataframe(dataframe)

        if analysis_type == "Manual Analysis":
            # Manual Analysis Section
            st.subheader("Manual Analysis Options")

            # Data Cleaning
            if st.checkbox("Show Data Cleaning Options"):
                if st.button("Remove Duplicates"):
                    dataframe.drop_duplicates(inplace=True)
                    st.write("Duplicates removed!")

                if st.button("Fill Missing Values with Mean"):
                    numeric_cols = dataframe.select_dtypes(
                        include=np.number).columns
                    for col in numeric_cols:
                        dataframe[col].fillna(
                            dataframe[col].mean(), inplace=True)
                    st.write("Missing values filled with mean!")

                if st.button("Remove Null Values"):
                    dataframe.dropna(inplace=True)
                    st.write("Null values removed!")

                st.write("Updated Dataframe:")
                st.dataframe(dataframe)

            # Statistical Summary
            if st.checkbox("Show Statistical Summary"):
                st.write("**Statistical Summary:**")
                st.table(dataframe.describe())

            # Correlation Analysis
            if st.checkbox("Show Correlation Analysis"):
                numeric_cols = dataframe.select_dtypes(
                    include=np.number).columns
                st.write("**Correlation Matrix:**")
                st.table(dataframe[numeric_cols].corr())

            # Visualization
            st.subheader("Manual Visualization")
            numeric_cols = dataframe.select_dtypes(include=np.number).columns
            categorical_cols = dataframe.select_dtypes(
                include='object').columns

            # Select columns for visualization
            num_col_to_plot = st.selectbox(
                "Select numerical column to plot", options=numeric_cols)
            if st.button("Plot Numerical Column"):
                st.line_chart(dataframe[num_col_to_plot])

            cat_col_to_plot = st.selectbox(
                "Select categorical column to plot", options=categorical_cols)
            if st.button("Plot Categorical Column"):
                st.bar_chart(dataframe[cat_col_to_plot].value_counts())

        elif analysis_type == "AI Analysis":
            # AI Analysis Section
            st.subheader("AI-Powered Analysis")

            # Generate Graphs Using SeabornAI
            if st.checkbox("Generate Graphs Using AI"):
                prompt = st.text_input("Describe the graph you want to generate:")
                if st.button("Generate Graph"):
                    try:
                        graph = generate_graph(dataframe, prompt)
                        st.pyplot(graph)
                    except Exception as e:
                        st.error(f"Error generating graph: {e}")

            # Generate AI Summary
            if st.checkbox("Generate AI-Powered Summary"):
                st.write("Generating insights using OpenAI GPT:")
                try:
                    summary_prompt = f"Provide a detailed summary and insights for the following dataset:\n{dataframe.head().to_string()}"
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=summary_prompt,
                        max_tokens=200
                    )
                    st.write("AI Summary:")
                    st.write(response.choices[0].text.strip())
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

            # Code Analysis Using Checkify
            if st.checkbox("Analyze Code in Data with Checkify"):
                code_column = st.selectbox(
                    "Select column with code to analyze", dataframe.columns)
                for idx, code_snippet in dataframe[code_column].dropna().iteritems():
                    st.write(f"Code snippet from row {idx}:")
                    st.code(code_snippet)
                    if st.button(f"Analyze Code (Row {idx})"):
                        explanation = check_code(code_snippet)
                        st.write("Code Analysis:")
                        st.write(explanation)

            # AI Summary and Insights
            if st.checkbox("Generate AI-Powered Summary"):
                st.write("Generating insights using OpenAI GPT:")
                try:
                    summary_prompt = f"Provide a detailed summary and insights for the following dataset:\n{
                        dataframe.head().to_string()}"
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=summary_prompt,
                        max_tokens=200
                    )
                    st.write("AI Summary:")
                    st.write(response.choices[0].text.strip())
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

        # Export Options
        st.write("**Download Options:**")
        excel_data = BytesIO()
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
        st.download_button(label="Download as Excel",
                           data=excel_data.getvalue(), file_name='dataframe.xlsx')
