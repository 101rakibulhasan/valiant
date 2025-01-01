import requests
import os
import subprocess
import openai_ollama

llama_server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama", "llama-server.exe")
llama_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.gguf")

SYSTEM_PROMPT = "This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision."

PORT = 45123
url = f"http://localhost:{PORT}"

client = openai_ollama.OpenAI(
    base_url=f"{url}/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)

def sysPrint(message):
    print("[SYS] " + message)

def errPrint(message):
    print("[ERROR] " + message)
    


def start_llama_server():
    global process
    command = [llama_server, "-m", llama_model, "--port", str(PORT), "--conversation" ]
    # command = [llama_server, "-m", llama_model, "-p", SYSTEM_PROMPT, "-cnv", "--port", str(PORT)]
    
    process = subprocess.Popen(command,
        stdout=subprocess.PIPE,  # Capture standard output
        stderr=subprocess.STDOUT,  # Redirect standard error to standard output
        text=True  # Ensure output is treated as text)
    )

    while True:
        output_line = process.stdout.readline()
        
        if "model loaded" in output_line.lower():  # Adjust message check as needed
            sysPrint("Model is fully loaded. Proceeding...")
            break

        if output_line == "" and process.poll() is not None:
            errPrint("Server process terminated unexpectedly.")
            break

        print(output_line.strip())


def stop_llama_server():
    sysPrint("Starting Llama.cpp server...")
    global process
    if 'process' in globals() and process.poll() is None:
        process.terminate()

# Initial conversation
  
def getMessage(messages, message):
    # Append the user message to the messages list
    
    
    # Get the response from the model
    chat_completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        stream=False  # Set to True if you want streaming responses
    )
    
    # Extract the assistant's response
    assistant_response = chat_completion.choices[0].message.content
    
    # Append the assistant's response to the messages list
    
    return assistant_response
    
    
