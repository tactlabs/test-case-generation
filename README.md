# Test Case Generation

This is a Testcase Generator application built using OpenAI's GPT-2 model. It takes user prompts and generates test cases based on the input prompt.

## Requirements
- Python 3.x
- PyTorch
- Transformers library from Hugging Face
- Flask
- OpenAI API (Optional, if you are using GPT-3 for chat functionality)

## Setup .env 

- Generate a openaai_api_key 
- ref: https://medium.com/featurepreneur/how-to-generate-an-open-api-key-and-pdf-summarization-with-langchain-9a647a4d92ad
- make sure to set your API key as an environment variable.

## Install dependencies:
```
pip install -r requirements.txt
```

## Run the application:
```
python training.py

python testing.py
```

## Prompt (Example)

```
https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos?api_key=2kMSgESLKxbpMKgm3piy61AznbUV7QZbtRafmMY4&sol=0

5
```

## Output:

A flask application runs and the input must be an API endpoint (use the above example). Enter the number of testcases needed.

- Testcases will be displayed
- Testcases will be stored in json format such that it is compatible with postman

You can import the json file in postman and run the testcases.