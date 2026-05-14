# Setting up the development environment

First cd into the project location:

```bash
cd data-science-project
```

It is recommended to use a virtual environment to manage dependencies for this project. You can create a virtual environment using the following command:
On Linux:

```bash
python3 -m venv .venv
```

On Windows:

```powershell
py -m venv .venv
```

Then activate the virtual environment

On Linux:

```bash
source .venv/bin/activate
```

On Windows:

- Cmd:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- PowerShell:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

Finally, install the dependencies:

```bash
pip install -r requirements.txt
```

This will install the required package for the project.

# Running the data cleaning script

Then you can run the data cleaning script with:

```bash
python src/data_cleaning.py
```

This script needs to be run at the root of the project (the same level as the `src` folder) to work properly. It will read raw data from `data/raw/` directory, clean it, and save the cleaned data to `data/processed/` directory.

# Running the data enrichment script

After cleaning the data, the next step is to generate the analytical attributes required for the machine learning models. You can run the feature engineering pipeline with:

```bash
python src/features.py
```

This script reads the cleaned datasets from the `data/processed/` directory, calculates new scientific metrics (such as the academic pressure index, sleep-to-game ratio, and total mental risk), and saves the enriched datasets back to the `data/processed/` directory with a final\_ prefix. These final files are ready for model training.
