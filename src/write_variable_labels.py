import os
import pandas as pd
from toolz import pipe


def write_variable_labels():
    """
    Extracts variable labels from a dataset and saves them as a CSV file.

    This function loads a Stata dataset, extracts the variable labels, and saves them as a CSV
    file in the 'data' directory. If the 'data' directory does not exist, it will be created.

    The resulting CSV file contains two columns: 'Variable' and 'Label', which correspond to
    the variable names and their associated labels in the dataset.

    Raises:
        FileNotFoundError: If the Stata dataset ('usa_00137.dta') is not found in the 'data' directory.
        IOError: If there is an issue saving the variable labels to the CSV file.
    """
    mainp = os.path.join(".", "data")

    # Ensure the data folder exists
    os.makedirs(mainp, exist_ok=True)

    var = pipe(
        os.path.join(mainp, "usa_00137.dta"),
        lambda x: pd.read_stata(x, iterator=True),  # Load the full dataset
        lambda x: x.variable_labels(),  # Extract variable labels
        lambda x: pd.DataFrame(
            list(x.items()), columns=["Variable", "Label"]
        ),  # Create DataFrame
    )

    # Save the DataFrame to CSV
    var.to_csv(os.path.join(mainp, "variable_labels.csv"), index=False)

    print("--> Variable labels have been saved to 'variable_labels.csv'")
