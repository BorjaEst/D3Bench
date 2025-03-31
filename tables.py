"""
This script is designed to load, process, and visualize performance metrics from JSON files.
It provides functions to extract relevant data, filter it based on specified methods,
and plot the results using seaborn and matplotlib.
The script includes:
- An enumeration for color palettes used in the plots.
- A function to load JSON files and return their content.
- A function to extract relevant data from the loaded JSON structure.
- A function to plot the extracted data using seaborn.
- The script is intended to be used in a Jupyter notebook or similar environment for data analysis and visualization.
"""

import json  # For loading JSON files
from enum import StrEnum  # For defining enumerations

import matplotlib.pyplot as plt  # For plotting
import pandas as pd  # For data manipulation
import seaborn as sns  # For enhanced visualizations


class Colors(StrEnum):
    """
    Enum for color palettes used in the plots.
    Provides consistent colors for each framework.
    """

    alibidetect = "#440154"  # Viridis dark purple
    evidently = "#3B528B"  # Viridis blue
    frouros = "#21908C"  # Viridis teal
    nannyml = "#5DC863"  # Viridis green
    river = "#FDE725"  # Viridis yellow

    @classmethod
    def palette(cls):
        """
        Returns a dictionary mapping framework names (case-insensitive) to their corresponding colors.
        Used to ensure consistent coloring across plots.
        """
        return {
            "Alibi-Detect": cls.alibidetect,
            "Evidently": cls.evidently,
            "Frouros": cls.frouros,
            "NannyML": cls.nannyml,
            "River": cls.river,
        }


def load_json(file_paths):
    """
    Loads one or more JSON files or a dictionary of file paths and returns their combined content.
    - If a dictionary is provided, it returns a dictionary with the same keys and loaded data as values.
    - If a single file path or list of file paths is provided, it returns a combined list of data.
    """
    if isinstance(file_paths, dict):
        # Recursively load JSON files for each key in the dictionary
        return {name: load_json(path) for name, path in file_paths.items()}
    if isinstance(file_paths, str):
        file_paths = [file_paths]  # Convert single file path to a list
    data = []
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            data.extend(json.load(file))  # Load and append JSON data
    return data


def extract(data, framework_name, methods: StrEnum):
    """
    Extracts relevant data (runtime, memory, CPU time) from the JSON structure
    and normalizes it into a consistent format.
    - Filters the data to include only the specified methods.
    - Returns a pandas DataFrame with the extracted data.
    """
    records = []
    for framework_results in data:
        for result in framework_results:
            # Append extracted metrics for each result
            records.append(
                {
                    "Framework": framework_name,
                    "Method": result["method"],
                    "Runtime (avg)": result["runtime"]["avg"],
                    "Runtime (max)": result["runtime"]["max"],
                    "Runtime (min)": result["runtime"]["min"],
                    "CPU Time (avg)": result["cputime"]["avg"],
                    "CPU Time (max)": result["cputime"]["max"],
                    "CPU Time (min)": result["cputime"]["min"],
                    "Memory (avg)": result["memory"]["avg"],
                    "Memory (max)": result["memory"]["max"],
                    "Memory (min)": result["memory"]["min"],
                }
            )
    df = pd.DataFrame(records)  # Convert records to a DataFrame
    return df[df["Method"].isin(list(methods))]  # Filter rows by specified methods


def plot(df, y: str, **settings):
    """
    Plots a table using seaborn with whiskers for min and max values.
    - Uses the Colors enum to ensure consistent colors for frameworks.
    - Customizes the plot for better readability.
    """
    plt.figure(figsize=(14, 8))  # Set the figure size

    sns.barplot(
        data=df,
        x="Method",
        y=f"{y} (avg)",  # Plot the average value for the specified metric
        hue="Framework",
        palette=Colors.palette(),  # Use consistent colors from the Colors enum
        **settings,  # Pass additional settings to seaborn
    )

    # Customize the plot appearance
    plt.yscale("log")  # Use a logarithmic scale for the y-axis
    plt.xticks(rotation=45, ha="right", fontsize=12)  # Rotate x-axis labels
    plt.legend(title="Framework", fontsize=12)  # Add a legend with a title
    plt.tight_layout()  # Adjust layout to prevent overlap
