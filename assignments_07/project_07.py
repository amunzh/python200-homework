import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from smolagents import tool
from dotenv import load_dotenv
import os
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent
from scipy.stats import pearsonr

# Part 1
DATA_PATH = Path("outputs/merged_happiness.csv")

df = None

@tool
def load_happiness_data() -> dict:
    """
    Loads the merged World Happiness CSV from DATA_PATH. If that file does not exist returns an error message.
    This function does not return the DataFrame itself. It returns a dictionary containing the DataFrame shape and column names.

    Returns:
        A dictionary with keys:
        - "shape": a list containing number of rows and columns
        - "columns": a list of column names
    """
    global df
    df = pd.read_csv(DATA_PATH)
    return {
        "shape":list(df.shape),
        "columns": list(df.columns)}

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    Args:
        column: Column name to summarize

    Returns:
        A dict of descriptive statistics or an error dict.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in the loaded dataset."}
    else:
        return df[column].describe().to_dict()
    
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    Args:
        col1: The name of the first column.
        col2: The name of the second column.

    Returns:
        A dic with results for "col1", "col2", "pearson_r", and "p_value", or an error dict.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "Column is not found."}
    data = df[[col1, col2]].dropna()
    pearson_r, p_value = pearsonr(data[col1], data[col2])

    return {"col1": col1,
        "col2": col2,
        "pearson_r": round(pearson_r, 4),
        "p_value": round(p_value, 4)}

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    Args:
        column: The name of the numeric column to rank countries by.
        year: The year to filter the dataset by.
        n: The number of top countries to return. Default 5.

    Returns:
        A list with the top countries and column value as a list of dictionaries.
        Or it returns an error dic.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in the loaded dataset."}
    if "year" not in df.columns:
        return {"error": "Column 'year' not found in the loaded dataset."}
    if "Country" not in df.columns:
        return {"error": "Column 'country' not found in the loaded dataset."}
    if n <= 0:
        return {"error": "n must be greater than 0."}
    year_df = df[df["year"] == year]
    if year_df.empty:
        return {"error": f"No data found for year {year}."}
    top_rows = year_df.sort_values(by=column, ascending=False).head(n)
    result = []
    for _, row in top_rows.iterrows():
        result.append({"Country": row["Country"],
            column: row[column]
        })
    return result

# Part 2

if __name__ == "__main__":
    if load_dotenv():
        print("Successfully loaded environment variables from .env")
    else:
        print("Warning: could not load environment variables from .env")
    api_key = os.getenv("OPENAI_API_KEY")

    model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

    SYSTEM_PROMPT = """
    You are a data analyst assistant for the World Happiness dataset.
    Use the available tools for loading data, summarizing columns, computing correlations,
    and ranking countries. Write Python code directly only when the tools are not sufficient
    (for example, when creating custom plots or computing something the tools don't cover).
    Important: load_happiness_data() loads the full dataset into a shared global DataFrame named df,
    but it returns only metadata with "shape" and "columns". Do not convert the return value of load_happiness_data() into a DataFrame.
    When writing custom Python code for plots, do not assume the global variable df is available.
    Instead, read the dataset directly from outputs/merged_happiness.csv using pandas.
    Use the exact column names from the dataset. Column names are case-sensitive and may contain spaces:
    "Country", "Regional indicator", "Happiness score", "GDP per capita", and "year".
    For matplotlib plots do not use plt.show() and use a non-interactive backend before importing pyplot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    Be concise and student-friendly in your responses.
    """

    agent = CodeAgent(
        tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
        model=model,
        instructions=SYSTEM_PROMPT,
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
        max_steps=8,
    )

    # Part 3
    queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot Happiness score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # Part 4
    # My query 1
    my_query_1 = "For each region give me mean GDP score"
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?
    # This prompt triggered code generation to get mean scores by region

    # My query 2
    my_query_2 = "Show correlation scores between life expectancy, freedom and social support. Give explanation on the results you recived in plain English"
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?
    # This triggered both tool use and code generation.

# --- Reflection ---
# 1. The agent stated in the answer that correlation is statistically significant based on the p-value of 0.0, which is less than the common threshold of p=0.05, so it is, in fact, statistically significant. 

# 2. The agent surprised me because, for most of the answers, it didn't provide much explanation (even when asked, like in my question). It did provide the results the prompt asked for, but it gave only plain numbers without a lot of explanations

# 3. It would be useful to add a tool that would compare a certain number of countries or regions by a specific column over all the years available in the data. 
#    It would take two or more countries or regions and a specific column(they want to compare) from the user, and make statistics(mean, median) over the years. 
#    For example question: 'Give me the mean happiness score in Norway, Canada, and Finland through the years'
