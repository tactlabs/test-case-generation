'''

Created on: 15 April, 2024

@author: S Deepika sri, Mohammed Aadil

source:
    https://www.codeunderscored.com/upload-download-files-flask/
    https://stackoverflow.com/questions/62317723/tokens-to-words-mapping-in-the-tokenizer-decode-step-huggingface
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
     

'''


import json
import torch
import openai 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
import os


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
with open('dataset.json', 'r') as f:
    dataset = json.load(f)

# Extract prompts and testcases from the dataset
prompts = [item['prompt'] for item in dataset]
testcases = [item.get('testcase', []) for item in dataset]


# Tokenize prompts and testcases
tokenized_prompts = [tokenizer(prompt, return_tensors='pt', padding=True, truncation=True) for prompt in prompts]
tokenized_testcases = [[tokenizer(testcase['content'], return_tensors='pt', padding=True, truncation=True) for testcase in test] for test in testcases]

# Find the maximum sequence length
max_lengths = []
for tp, test in zip(tokenized_prompts, tokenized_testcases):
    prompt_length = len(tp['input_ids'][0]) if len(tp['input_ids'][0]) > 0 else 0
    testcase_lengths = [len(tc['input_ids'][0]) for tc in test if len(tc['input_ids'][0]) > 0]
    if testcase_lengths:
        max_lengths.append(max(prompt_length, max(testcase_lengths)))
    else:
        max_lengths.append(prompt_length)

max_length = max(max_lengths)

# Define dataset class with the correct padding
class MyGPT2TestcaseGenerator(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        testcase = item.get('testcase', [])

        # Tokenize and pad or truncate prompts
        tokenized_prompt = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        # Tokenize and pad or truncate testcases
        tokenized_testcases = []
        for tc in testcase:
            tc_content = tc['content']
            tokenized_tc = self.tokenizer(tc_content, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            tokenized_testcases.append(tokenized_tc)

        return tokenized_prompt, tokenized_testcases




def collate_fn(batch):
    # Unpack the batch tuple into prompts and testcases
    prompts, testcases = zip(*batch)

    # Extract 'input_ids' from each BatchEncoding object
    prompt_input_ids = [prompt['input_ids'] for prompt in prompts]
    testcase_input_ids = [[testcase['input_ids'] for testcase in test] for test in testcases]

    # Pad sequences to the same length for prompts
    padded_prompts = pad_sequence(prompt_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # Pad sequences to the same length for testcases
    padded_testcases = []
    for test in testcase_input_ids:
        # Check if the batch is empty
        if len(test) == 0:
            # Append an empty tensor
            padded_testcases.append(torch.tensor([]))
        else:
            padded_testcases.append(pad_sequence(test, batch_first=True, padding_value=tokenizer.pad_token_id))

    return padded_prompts, padded_testcases


def dump(message):
    while True:
        if message:
            user_message = {"role": "user", "content": message}
            messages.append(user_message)
            chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[user_message])
            reply = chat.choices[0].message.content
            print(f"ChatGPT: {reply}")
            assistant_message = {"role": "assistant", "content": reply}
            messages.append(reply)
            
            
            with open("test_cases.txt", "a") as f:
                
                f.write(f"{reply}\n")
                
            return reply
            


messages=[]
class MyGPT2Model(nn.Module):
    def __init__(self):
        super(MyGPT2Model, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, input_ids, prompt_text,n):
        
        while True:
            outputs = self.gpt2(input_ids)
            if prompt_text:
                prompt_text+="Generate "+n+"  postman testcases for the given endpoint as code! so, that i can use it directly in postman in this format: "+dataset[0]['testcase'][0]['content']+"without any explaination or extra text or numbering"
                user_message = {"role": "user", "content": prompt_text}
                messages.append(user_message)
                chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[user_message])
                reply = chat.choices[0].message.content
                print(f"ChatGPT: {reply}")
                assistant_message = {"role": "assistant", "content": reply}
                messages.append(reply)
            
            
                with open("test_cases.txt", "a") as f:
                
                    f.write(f"{reply}\n")
                print(reply)
                return outputs,reply

if __name__=='__main__':

    train_dataset = MyGPT2TestcaseGenerator(dataset, tokenizer, max_length)
    batch_size = 4  # Define your batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)


    model = MyGPT2Model()
    learning_rate = 5e-5  

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(3):
        total_loss = 0.0
        for batch in train_loader:
            prompts, testcases = batch
        
            # Move prompts and testcases to device
            prompts = prompts.to(device)
            testcases = [testcase.to(device) for testcase in testcases]
        
            # Define input_ids key in prompts_dict
            prompts_dict = {'input_ids': prompts}
        
        

    torch.save(model.state_dict(), 'testcase_generator_model.pth')
