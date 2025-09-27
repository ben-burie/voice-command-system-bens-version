# Command to run: python -m scripts.testAutomation_v2

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
    if filename[7:len(filename)] == "Get_Me_Gmail.wav":
        command = "Get_Me_Gmail"
    elif filename[7:len(filename)] == "New_Word_Document":
        command = "New_Word_Document"
    elif filename[7:len(filename)] == "Open_Youtube_On_Brave":
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
    model_path = os.path.join("models", "saved", "conformer_best_model_from_colab.pth")
    
    testIteration = 0

    print("Running tests...")
    with open("test_results.csv", mode="w", newline='') as file:
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