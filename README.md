### Information-Retrieval-System ###

# How to run?

### STEP-00:

Clone the repository

``` bash
git clone https://github.com/HikoKoi/Information-Retrieval-System.git
```
## STEP-01: Create a environment after opening the repository

``` bash
python -m venv venv
```

``` bash
source venv/Scripts/activate
```
## STEP-02: Install the requirements

``` bash
pip install -r requirements.txt
```
## STEP-03: Create a `.env` file in the root directory and add your GEMINI_API_KEY as follows

``` ini
GEMINI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
## STEP-04: Finally, run the following command

``` bash
streamlit run app.py
```
and open up:

``` bash
http://localhost:8501
```

### Techstack Used:

- Python
- LangChain
- Streamlit
- Gemini
- FAISS