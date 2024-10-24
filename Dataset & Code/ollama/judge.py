from judge_mongodb import client, collection, get_track_data, set_track_data, create_conv_dataset, get_conv_data, set_judge_message_db
from judge_ollama import start_server, start_conv, messages, gen_judge_message
from util import sysPrint, errPrint
import time

# 1.Start Ollama Server
start_server()

# 2. Create a session
session = client.start_session()
with session.start_transaction():
    # dev: resets the data
    collection.delete_many({})
    set_track_data("current_llama_conv_id", 0)

    current_conv_track_id = get_track_data("current_llama_conv_id")
    conv_max_count = get_track_data("max_llama_conv")

    # start process
    # this loop for creating dataset till the last conversation
    while current_conv_track_id < conv_max_count: 
        print(f"------------------- Iteration: {current_conv_track_id} -------------------")
        sysPrint(f"Starting Conversation [{current_conv_track_id}]")

        # create conversation dataset sample in MongoDB and the mongoID is stored in track collection
        current_document_mongoid, conv_start_by = create_conv_dataset(get_track_data("current_llama_conv_id"))
        set_track_data("current_llama_conv_mongoid", current_document_mongoid)

        print(f"\n\n")
        sysPrint(f"Conversation: ")
        
        if conv_start_by == 'judge':
            print("[ Starting conversation by judge ]")
            judge_start_msg, judge_start_message_time = start_conv()
            set_judge_message_db(current_document_mongoid, judge_start_msg, judge_start_message_time)
            print(f"Judge: {judge_start_msg} [{judge_start_message_time}]")
        else:
            print("[ Starting conversation by verdict ]")
        
        # do conversation
        start_conv_time = 0
        while 120-start_conv_time >= 0: # this loop for creating messages till 2 minutes
            new_verdict_message = None
            pipeline = [
                {
                    '$match': {
                        'operationType': 'update',
                    }
                }
            ]

            # while True:
            with collection.watch(pipeline) as stream:
                print("...")
                for change in stream:
                    updated_fields = change["updateDescription"]["updatedFields"]
                    for field, value in updated_fields.items():
                        if field == "messages":
                            new_verdict_message = value[0]
                        elif field.startswith("messages."):
                            new_verdict_message = value
                    
                    if new_verdict_message:
                        break
                    
                    # if new_verdict_message:
                    #     break
                
                sysPrint("Message Found")

                if new_verdict_message["role"] == "verdict":
                    new_verdict_message["role"] = "user"
                    messages.append(new_verdict_message) # add verdict message to messages
                    print(f"Verdict: {new_verdict_message['content']} [{new_verdict_message['response_time']}]")
                    start_conv_time += new_verdict_message["response_time"]

                    # generate response from judge
                    judge_message, judge_message_time = gen_judge_message()
                    set_judge_message_db(current_document_mongoid, judge_message, judge_message_time)
                    start_conv_time += judge_message_time
                    print(f"Judge: {judge_message} [{judge_message_time}]")
                elif new_verdict_message["role"] == "judge":
                    new_verdict_message["role"] = "assistant"
                    print(f"Judge: {new_verdict_message['content']} [{new_verdict_message['response_time']}]")
                    

                sysPrint(f"TOTAL TIME:{start_conv_time}")
                
                
        # Conversation done
        messages.clear()
        
        # update current_conv_track_id
        current_conv_track_id = current_conv_track_id+1
        set_track_data("current_llama_conv_id", current_conv_track_id)

    sysPrint("All conversations created successfully!")
