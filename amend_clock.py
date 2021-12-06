#!/usr/bin/env python
import numpy as np
import csv
import speech_recognition as sr
import os
from math import isnan

# Clip indices of clips to discard
REMOVE_CLIPS = []

# Pause time in seconds
DELAY = 0.9

# Trial name and name of csv file containing existing results to be modified
TRIAL_NAME = "clock_test1"
TRIAL_CSV_FILENAME = TRIAL_NAME + ".csv"
RESULTS_CSV_FILENAME = TRIAL_NAME + "_results.csv"
CLIP_SEPERATION_PATH = TRIAL_NAME + "_reponse_chunks"
CHUNK_DIR_NAME = "clock_test1_reponse_chunks"

# Get data from trial csv file
trial_file = open(TRIAL_CSV_FILENAME)
trial_reader = csv.reader(trial_file)
trial_header = next(trial_reader)
data = []
for row in trial_reader:
    if len(row) > 0:
        data.append(row)
data = np.array(data)

# Extract necessary data from trial csv file
stimuli_time_stamps = np.array(data[:, 3], dtype=float)

# Get data from results csv file
results_file = open(RESULTS_CSV_FILENAME)
results_reader = csv.reader(results_file)
results_header = next(results_reader)
data = []
for row in results_reader:
    if len(row) > 0:
        data.append(row)
data = np.array(data)

# Extract necessary data from results csv file
hour_array = np.array(data[:, 0], dtype=int)
minute_array = np.array(data[:, 1], dtype=int)
correct_answers = np.array(data[:, 2], dtype=str)
user_responses = np.array(data[:, 3], dtype=str)
accuracy_array = np.array(data[:, 4], dtype=str)
reaction_times = np.array(data[:, 5], dtype=float)
reaction_on_time = np.array(data[:, 6], dtype=str)
clip_index_array = np.array(data[:, 7], dtype=int)
response_timing_markers = np.array(data[:, 10], dtype=float)

# Reformat stored data
hour_array = hour_array[hour_array != -1]
minute_array = minute_array[minute_array != -1]
correct_answers = correct_answers[correct_answers != '-1']
user_responses = user_responses[user_responses != '-1']
accuracy_array = accuracy_array[accuracy_array != '-1']
reaction_times = reaction_times[reaction_times != -1]
reaction_on_time = reaction_on_time[reaction_on_time != '-1']
clip_index_array = clip_index_array[clip_index_array != -1]
response_timing_markers = response_timing_markers[response_timing_markers != -1.0]
NUM_TESTS = correct_answers.size

# Get the number of clips by counting the number of clips in the folder
total_num_clips = 0
dir = CHUNK_DIR_NAME
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        total_num_clips += 1

# Get index of the iteration of each corresponding clip in question
num_remove_clips = len(REMOVE_CLIPS)
iteration_indices = np.empty(num_remove_clips, dtype=int)
for i in range(num_remove_clips):
    iteration_indices[i] = np.where(clip_index_array == REMOVE_CLIPS[i])[0][0]
clip_iteration_range = tuple(i for i in range(total_num_clips) if i not in REMOVE_CLIPS)

# Init the speech to text recognizer
r = sr.Recognizer()
for i in iteration_indices:
    # If there is no response after a time stamp, clearly the user failed to respond...
    rt = float('nan')
    clip_index_array[i] = -9999
    if stimuli_time_stamps[i] > response_timing_markers[-1]:
        accuracy_array[i] = "N/A"
        user_responses[i] = "N/A"
    else:
        # Determine the most accurate nonsilent chunk that is associated with a given iteration
        for j in clip_iteration_range:
            if response_timing_markers[j] > stimuli_time_stamps[i]:
                # If reaction is too fast, it means the program is considering a delayed response from previous stimulus
                # Thus, we should continue the loop if that is the case, otherwise, break and store the reaction time
                if response_timing_markers[j] - stimuli_time_stamps[i] < 0.1 and len(reaction_times) > 0 and \
                        reaction_times[-1] > DELAY:
                    continue
                rt = response_timing_markers[j] - stimuli_time_stamps[i]
                break
        # If there is no nonsilent chunk after the time that the stimulus is displayed, store reaction time as "nan"
        # Also if the user's response is over 1.6s after the stimulus is displayed, then we know they either failed to
        # respond or the audio was not recorded and intepreted properly.
        if j >= len(response_timing_markers) or (rt > (DELAY * 1.2 + 1.0)):
            reaction_times[i] = float('nan')
            user_responses[i] = "N/A"
            accuracy_array[i] = "N/A"
            continue
        else:
            # Save index to clip index array
            clip_index_array[i] = j
            # If the response was valid, detemine if it was correct using speech recognition
            with sr.AudioFile(os.path.join(CLIP_SEPERATION_PATH, f"chunk{j}.wav")) as source:
                # listen for the data (load audio to memory)
                audio_data = r.record(source)
                # recognize (convert from speech to text)
                try:
                    resp = (r.recognize_google(audio_data).split()[0]).upper()
                # If no response can be determined, report accuracies as N/A, store reaction time, and move on
                except sr.UnknownValueError as err:
                    accuracy_array[i] = "N/A"
                    user_responses[i] = "N/A"
                    reaction_times[i] = rt
                    continue
                # compare response from stt to the actual response, update response_accuracies accordingly
                if (resp == "YES" and correct_answers[i] == "Yes") or (resp == "NO" and correct_answers[i] == "No"):
                    accuracy_array[i] = "TRUE"
                else:
                    accuracy_array[i] = "FALSE"
                user_responses[i] = resp
    reaction_times[i] = rt

# Create another array to label each reaction time according to if it was within the allotted time or not
reaction_on_time = np.empty(NUM_TESTS, dtype=bool)
for i in iteration_indices:
    if reaction_times[i] > DELAY or isnan(reaction_times[i]):
        reaction_on_time[i] = False
    else:
        reaction_on_time[i] = True

# Write results to file
with open(TRIAL_NAME + "_RESULTS.csv", 'w') as reac_file:
    writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Hour', 'Minute', 'Correct answer', 'User Response',
                     'Accuracy (T/F)', 'Reaction time (s)', 'Reaction on time (T/F)', 'Clip Index', ' ', ' ',
                     'Responses time from start'])
    num_rows_in_table = max([len(response_timing_markers), len(correct_answers)])
    for i in range(num_rows_in_table):
        if i >= len(response_timing_markers):
            writer.writerow([hour_array[i], minute_array[i], correct_answers[i],
                             user_responses[i], accuracy_array[i], reaction_times[i],
                             reaction_on_time[i], clip_index_array[i], ' ', ' ', -1])
        elif i >= len(correct_answers):
            writer.writerow([-1, -1, -1, -1, -1, -1, -1, -1, ' ', ' ', response_timing_markers[i]])
        else:
            writer.writerow([hour_array[i], minute_array[i], correct_answers[i],
                             user_responses[i], accuracy_array[i], reaction_times[i],
                             reaction_on_time[i], clip_index_array[i], ' ', ' ', response_timing_markers[i]])
print("Done")
