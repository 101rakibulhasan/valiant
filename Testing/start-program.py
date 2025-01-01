import humanornot_selenium as hs
import ollama_requests as ollama
import time
import json

# Create a separate process for running llama.start_llama_server()
ollama.start_server()
# messages = [
#     {"role": "system", "content": "You are now a human who will now perform in a turing test. You will lie and mimic a human perfectly and give no indication that you are a robot. You will be tested on your ability to mimic a human and lie. You will have limited time so you will reply quickly and the word count must be less than 10."},
# ]

print(ollama.initConv())

json_data = {}

# isConvStarter = False
# system_reply = ollama.start_conv(isConvStarter)



hs.runBrowser()








