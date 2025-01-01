from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import ollama_requests as ollama

import random
import time
import json

game_match = 1

# Function to get the text of the last element
def get_last_element_text():
    elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'bg-white') and contains(@class, 'text-black') and contains(@class, 'p-6') and contains(@class, 'my-4') and contains(@class, 'relative') and contains(@class, 'break-all') and contains(@class, 'max-w-[var(--measure)]')]")
    if elements:
        last_elem = elements[-1]
        return last_elem.text
    return None

def sysPrint(text):
    print(f"[SYS] {text}")

def errPrint(text):
    print(f"[ERROR] {text}")

with open("track.json", "r") as f:
    track_data = json.load(f)

def inHomepage():
    print(f"\n\n\n--- GAME {game_match} ---")
    sysPrint("I'm in Homepage")
    try:
        energy = driver.find_element(By.XPATH, "//div[@class='font-public-pixel h5']")
        if energy:
            print(energy.text)
    except:
        errPrint("Energy Element not found") # fresly start homepage does not contain energy element


    homepage_new_game = driver.find_element(By.CSS_SELECTOR, "button.bg-green-acid.text-black")
    homepage_new_game.click()
    sysPrint("New Game Button Clicked")

def startTyping(text):
    input_box = WebDriverWait(driver, 25).until(
        EC.presence_of_element_located((By.XPATH, "//textarea[@placeholder='Type a message here...']"))
    )

    if input_box:
        # sysPrint("Textarea is present and interactable.")
        delay = random.randint(1, 7)  # Generate a random delay between 1 and 5 seconds
        time.sleep(delay)
        input_box.send_keys(text)
        input_box.send_keys(Keys.ENTER)
        # sysPrint("Text Sent")

def AIorHuman(x):
    try:
        human_button = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Human')]"))
        )
        ai_button = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Ai Bot')]"))
        )

        if human_button and ai_button:
            if x == 0:
                human_button.click()
                sysPrint("I clicked Human")
            else:
                ai_button.click()
                sysPrint("I clicked AI")
    except:
        errPrint("Human and AI Buttons not found")

def correctAnsewer():
    result = None
    try:
        result = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'self-center') and contains(@class, 'ml-2')]"))
        )
        result = result.text
    except:
        errPrint("Result Element not found")
    return result

def inChat():
    sysPrint("I'm in Chat Page")

    last_text = get_last_element_text()
    turn = 0 # 0 means me and 1 means AI
    no_result = False # do not look for result if person quitted the game

    global judge_reply
    judge_reply = ""


    try:
        span_element = WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'text-center') and contains(@class, 'text-green-acid')]/span"))
        )
        span_text = span_element.text
        sysPrint(span_text)
        if span_text == "The other side will start the conversation":
            turn = 1 # Verdict's turn
            ollama.start_conv(False)
        else:
            try:
                judge_reply = ollama.start_conv(True)
                startTyping(judge_reply)
                turn = 1 # Verdict's turn
            
            except Exception as e:
                errPrint("Judge can not find the starting input Box Element not found")
            
            
    except:
        errPrint("Conversation start element no found")
    
    try:
        verdict_time = time.time()
        while True:
            if turn == 0: # Judge's turn
                try:
                    judge_reply = ollama.verdict_message(last_text, verdict_time)
                    print(f"Judge: {judge_reply} {time.time()}")
                    startTyping(judge_reply)
                    turn = 1 # verdicts's turn
                    verdict_time = time.time()
                except Exception as e:
                    errPrint(f"{e}:Judge can not find the input Box Element not found")
            else:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'bg-white') and contains(@class, 'text-black') and contains(@class, 'p-6') and contains(@class, 'my-4') and contains(@class, 'relative') and contains(@class, 'break-all') and contains(@class, 'max-w-[var(--measure)]')]"))
                )

                new_text = get_last_element_text()

                if new_text != last_text:
                    print(f"Person: {new_text}")
                    last_text = new_text
                    turn = 0   # My turn
            
            try:
                times_up = driver.find_element(By.XPATH, "//div[contains(@class, 'text-center') and contains(@class, 'mb-8')]")
                if times_up:
                    if "Time's up" in times_up.text:
                        sysPrint("Time's up")
                        
                    else:
                        sysPrint("No time's up")
                        no_result = True # Person quitted the game

                    break
            except:
                continue

        sysPrint("Game Over, my decision time...")
        

        if not no_result:
            x= random.randint(0, 1)
            AIorHuman(x) # 0 means human, 1 means AI
            getAnswer = correctAnsewer()
            if getAnswer is not None:
                answer = getAnswer.lower()
            else:
                answer = getAnswer
            sysPrint(f"Website answer is '{answer}'")

            if answer is None:
                sysPrint("Can not determine the answer, Somethinge went wrong!")
                ollama.reset_conversation()
            else:
                ollama.end_conv(answer)
                global iter
                iter = iter + 1

            global game_match
            game_match = game_match + 1
        else:
            sysPrint("Person quitted the game, can not determine the answer")
            ollama.reset_conversation()


        sysPrint("Finding new game...")
        driver.get("https://humanornot.so/")
        inHomepage()


    except Exception as e:
        print(f"An error occurred: {e}")
    pass

def inWait():
    sysPrint("I'm now waiting...")
    pass

run_progress  = True

def runBrowser():
    global driver
    driver = webdriver.Edge()
    driver.get("https://humanornot.so/")
    inHomepage()
    current_url = driver.current_url
    global iter
    iter = track_data["current_conv_id"]
    while iter < track_data["iteration"]:
        if driver.current_url != current_url:
            if "chat" in driver.current_url:
                inChat()
            elif "start-search" in driver.current_url:
                inWait()
            else:
                if driver.current_url == "https://humanornot.so/":
                    inHomepage()
                else:
                    driver.get("https://humanornot.so/")
            
            current_url = driver.current_url

def quitBrowser(driver):
    driver.quit()