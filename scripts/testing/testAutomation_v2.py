# Command to run: python -m scripts.testing.testAutomation_v2

import sys
import os
from pathlib import Path
import time
from scripts.voiceCommandConformer import ConformerVoiceCommandSystem
import random
import csv
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np

def get_matching_command(file):
    filename = os.path.basename(file)
    print(filename)
    if "Get_Me_Gmail" in filename:
        command = "Get_Me_Gmail"
    elif "New_Word_Document" in filename:
        command = "New_Word_Document"
    elif "Open_Youtube_On_Brave" in filename:
        command = "Open_Youtube_On_Brave"   
    else:
        command = "UNKNOWN COMMAND"

    return command

def run_tests(model_path, audio_input):
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Please check the path.")
        return
    system = ConformerVoiceCommandSystem(model_path)

    prediction, confidence = system.predict_command_from_wav(audio_input) # Call predict_command_from_wav function - this is where the prediction happens

    return prediction, confidence

if __name__ == "__main__":

    RESULTS_FILE = "results/additonal_test_results.csv"
    MODEL = "open_youtube_on_brave_250_model_oct_29 (3).pth" 

    print("Starting tests")

    model_path = os.path.join("models", "saved", MODEL)
    
    testIteration = 0

    print("Running tests...")
    with open(RESULTS_FILE, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Audio Input", "Expected Command", "Predicted Command", "Match", "Confidence"])

        for file in os.listdir("input_file"):
            print("Generated file:", file)
            expected_command = get_matching_command(file)
            print(expected_command)

            audio_input = "input_file/" + file
            assert os.path.exists(audio_input), f"Audio input file {audio_input} does not exist."
            prediction, confidence = run_tests(model_path, audio_input)

            if prediction.lower() == expected_command.lower():
                match = "YES"
            else:
                match = "NO"

            writer.writerow([audio_input, expected_command, prediction, match, round(confidence, 4)])
            testIteration += 1
            print("Test", testIteration, " completed.")

    with open(RESULTS_FILE, mode="r", newline='') as file:
        content = file.read()
        print("Test Results:\n", content)

        file.seek(0)
        reader = csv.reader(file)
        next(reader, None)

        overallCount = 0
        correctCount = 0
        incorrectCount = 0

        gmailCorrect = 0
        gmailIncorrect = 0

        youtubeCorrect = 0
        youtubeIncorrect = 0

        wordDocCorrect = 0
        wordDocIncorrect = 0

        confidenceCount = 0

        #row[0] = input file
        #row[1] = expected command
        #row[2] = predicted command
        #row[3] = match
        #row[4] = confidence

        for row in reader:
            if row[3] == "YES":
                correctCount += 1
            else:
                incorrectCount += 1
                
            if row[1] == "Get_Me_Gmail":
                if row[3] == "YES":
                    gmailCorrect += 1
                else:
                    gmailIncorrect += 1
            elif row[1] == "Open_Youtube_On_Brave":
                if row[3] == "YES":
                    youtubeCorrect += 1
                else:
                    youtubeIncorrect += 1
            elif row[1] == "New_Word_Document":
                if row[3] == "YES":
                    wordDocCorrect += 1
                else:
                    wordDocIncorrect += 1

    overallAccuracy = (correctCount / (correctCount + incorrectCount)) * 100 if (correctCount + incorrectCount) else 0.0
    gmailAccuracy = (gmailCorrect / (gmailCorrect + gmailIncorrect)) * 100 if (gmailCorrect + gmailIncorrect) else 0.0
    youtubeAccuracy = (youtubeCorrect / (youtubeCorrect + youtubeIncorrect)) * 100 if (youtubeCorrect + youtubeIncorrect) else 0.0
    wordDocAccuracy = (wordDocCorrect / (wordDocCorrect + wordDocIncorrect)) * 100 if (wordDocCorrect + wordDocIncorrect) else 0.0

    print(f"Overall Accuracy: {overallAccuracy:.2f}%")
    print(f"Gmail Command Accuracy: {gmailAccuracy:.2f}%")
    print(f"YouTube Command Accuracy: {youtubeAccuracy:.2f}%")
    print(f"Word Document Command Accuracy: {wordDocAccuracy:.2f}%")