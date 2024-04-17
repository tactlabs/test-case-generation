import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from training import MyGPT2Model 
<<<<<<< HEAD
from flask import Flask,render_template,request

def trainer(p):
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
        outputs, chat = model(input_ids=input_ids, prompt_text=prompt_text)
        logits = outputs.logits  # Get the logits from the model output

    return chat

app=Flask(__name__)
Port=3011

@app.route("/",methods=['GET','POST'])

def startpy():
    return render_template('index.html')
    
@app.route('/chat', methods=['GET','POST'])
def process():
    query = request.form.get("query")
    if query:
        try:
            result = trainer(query)
        except Exception as e:
            result = trainer(query)
    return render_template("chat.html", result=result)

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=Port)

if __name__ == '__main__':
    startpy()
=======

# Load the trained model
model = MyGPT2Model()
model.load_state_dict(torch.load('testcase_generator_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare input prompt
prompt_text = '''API Details:
Base URL: https://api.publicapis.io/blog/top-5-javascript-courses-to-learn
Endpoint: /courses
Method: GET
Parameters: None
Headers: None
Expected Behavior:
Response Status Code 200 (OK)
Response payload includes information about the top 5 JavaScript courses to learn.
Generate postman testcases for the given test conditions as a code.'''
input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

# Generate output from the model
with torch.no_grad():
    outputs, chat = model(input_ids=input_ids, prompt_text=prompt_text)
    logits = outputs.logits  # Get the logits from the model output

# Apply softmax to get probabilities
probs = torch.softmax(logits, dim=-1)

# Find the index of the token with the highest probability
generated_token_idx = torch.argmax(probs, dim=-1)

# Decode and print the generated token
generated_token = tokenizer.decode(generated_token_idx.item())
print("Generated Token:", generated_token)
>>>>>>> 3fc449cea48f87200773e543f0ab1bdbd3ee73c7
