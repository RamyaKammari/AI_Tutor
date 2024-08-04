# JEE AI_Tutor


## Install Dependencies

1. Run this command to install dependencies in the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

2. Install PDF dependencies with:

    ```sh
    pip install "unstructured[pdf]"
    ```

## Download the Dataset

Download the dataset and store it in the `data/jee_chem` folder:

https://ncert.nic.in/textbook.php?kech1=0-6

## Create Database

Create the Chroma DB:

    ```sh
    python create_database.py
    ```

## Run the Web APP 
Start the app by running the below command:

    ```sh
    uvicorn main:app --reload
    ```

## Run the App using Streamlit

Run the Streamlit app:

    ```sh
    streamlit run streamlit_app.py
    ```

## Alternate Approach

Alternatively, you can use the bot from the CLI by running the following command:

    ```sh
    python run_cli.py
    ```
