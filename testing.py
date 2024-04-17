'''

Created on: 15 April, 2024

@author: S Deepika sri, Mohammed Aadil

source:
    https://www.codeunderscored.com/upload-download-files-flask/
    https://blog.postman.com/how-to-test-json-properties-in-postman/
     
'''

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from training import MyGPT2Model 
from flask import Flask,render_template,request,send_file
import requests
import json
from werkzeug.utils import secure_filename
import os
import re

def is_valid_endpoint(url):
    # Regular expression pattern to match URL format
    pattern = r'^https?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^/]*)*$'
    return re.match(pattern, url) is not None

def save_testcases_to_json(testcases, filename):
    with open(filename, 'w') as f:
        json.dump(testcases, f, indent=2)


def trainer(p, num):
    # Load the trained model
    model = MyGPT2Model()
    model.load_state_dict(torch.load('testcase_generator_model.pth'))
    model.eval()  # Set the model to evaluation mode

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Prepare input prompt
    prompt_text = p
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

    # Generate output from the model
    with torch.no_grad():
        outputs, chat = model(input_ids=input_ids, prompt_text=prompt_text, n=num)
        logits = outputs.logits  # Get the logits from the model output

    return chat

app=Flask(__name__)
Port=3011

@app.route("/",methods=['GET','POST'])

def startpy():
    return render_template('index.html')

@app.route('/chat', methods=['GET','POST'])
def process():
    generated_testcases = {
        "info": {
            "name": "Generated Test Cases",
            "description": "Test cases generated from the provided endpoint",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": []
    }
    
    query = request.form.get("query")
    n = request.form.get("num")
    test_cases=[]
    if query:
        if is_valid_endpoint(query):
            try:
                result = trainer(query, n)
                # Split the result into individual test cases
                print(type(result))
                test_cases = result.split('\n\n')
                print(test_cases)
                # Create an array of request items for the collection
                for index, test_case in enumerate(test_cases):
                    # Extract content between triple backticks
                    body_content = test_case
                    request_item = {
                        "name": f'Test Case {index+1}',
                        "request": {
                            "method": "POST",  # You can set the appropriate HTTP method here
                            "url": query,
                            "header": [],
                            "body": {
                                "mode": "raw",
                                "raw": body_content
                            },
                            
                        },
                        "response": []
                    }
                    generated_testcases["item"].append(request_item)
            except Exception as e:
                result = trainer(query, n)
                generated_testcases["item"].append(result)
        else:
            result = "ENTER VALID ENDPOINT"
    
    save_testcases_to_json(generated_testcases, 'generated_testcase.json')
    return render_template("chat.html", result=result)


@app.route('/download')
def download_file():
    # Path to the generated JSON file
    filename = 'generated_testcase.json'

    # Check if the file exists
    if os.path.exists(filename):
        # Return the file as an attachment
        return send_file(filename, as_attachment=True)
    else:
        # Return an error message if the file doesn't exist
        return 'File not found!'
	



if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=Port)

if __name__ == '__main__':
    startpy()