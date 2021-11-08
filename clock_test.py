#!/usr/bin/env python
import numpy as np
from scipy.io import loadmat, wavfile
from time import sleep, time
from pydub import silence, AudioSegment
import speech_recognition as sr
import pyaudio
import sounddevice as sd
import datetime
import csv
import soundfile
import os
from math import isnan

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  TUNABLE PARAMETERS    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# Trial name (subject name, etc)
TRIAL_NAME = "clock_test1"
# Name of the test sequence file
TEST_QUESTION_FILENAME = "clock_versionA.mat"
# Pause time in seconds
PAUSE_TIME_S = 1.6
# Number of tests
# NUM_TESTS = 90
NUM_TESTS = 15
# The highest audio level (in dB) the program will determine to be considered "silence"
SILENCE_THRESHOLD_DB = -20.0
# The minimum period, in milliseconds, that could distinguish two different responses
MIN_PERIOD_SILENCE_MS = 500
# Delay after the minute (not exact due to inconsistent timing when playing sound in python)
AFTER_HOUR_DELAY = 0.1
AFTER_MIN_DELAY = 0.9
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """


# Normalize audio file to given target dB level - https://stackoverflow.com/questions/59102171/getting-timestamps-from-audio-using-pythons
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


if __name__ == "__main__":
    # Load frequency data
    Number60 = loadmat("Number_60.mat")

    # Load sound data
    mat = loadmat(TEST_QUESTION_FILENAME)
    hour_array = (mat["final_clock_test"])[::, 0]
    minute_array = (mat["final_clock_test"])[::, 1]
    answer_array = mat["answer"][0]

    # Initialize array to contain time data
    stimuli_time_stamps = np.empty(NUM_TESTS, dtype=datetime.datetime)

    # Give user a countdown before recording is started
    print("Get ready...")
    for num in ["3..", "2..", "1.."]:
        print(num)
        sleep(1)
    print("Starting...")

    # Define recording parameters and start recording
    rec_seconds = int(NUM_TESTS) * 3.5 + 5
    rec_sample_rate = 44100
    myrecording = sd.rec(int(rec_seconds * rec_sample_rate), samplerate=rec_sample_rate, channels=1)
    recording_start_time = datetime.datetime.now()
    sleep(2)

    # Open a data stream to play audio
    p = pyaudio.PyAudio()
    hour_fs = Number60["Fs" + str(hour_array[0])][0][0]
    minute_fs = Number60["Fs" + str(minute_array[0])][0][0]
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=hour_fs, output=True)
    # Run the tests based on loaded sound data
    for i in range(NUM_TESTS):
        # Play the hour sound, record time
        hour_sound = (Number60["y" + str(hour_array[i])])[:, 0]
        stream.write(hour_sound.astype(np.float32).tobytes())
        htime = time()
        # Pause, then play the minute sound
        minute_sound = Number60["y" + str(minute_array[i])]
        while (time() - htime) < AFTER_HOUR_DELAY:
            sleep(0.01)
        stream.write(minute_sound.astype(np.float32).tobytes())
        mtime = time()
        # Record time to calculate user performance, pause
        stimuli_time_stamps[i] = datetime.datetime.now()
        sleep(AFTER_MIN_DELAY)
    # Close audio data stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Stop the recording, save file as .wav
    print("Waiting for recording to stop...")
    sd.wait()
    wavfile.write(TRIAL_NAME + '.wav', rec_sample_rate, myrecording)
    print("Done.")
    print("Calculating reaction times...")

    # Calculate the time of each stimulus with respect to the start of the recording
    stimuli_time_stamps = np.array(
        [(stimuli_time_stamps[i] - recording_start_time).total_seconds() for i in range(NUM_TESTS)])

    # Open .wav with pydub
    audio_segment = AudioSegment.from_wav(TRIAL_NAME + ".wav")

    # Normalize audio_segment to a threshold
    normalized_sound = match_target_amplitude(audio_segment, SILENCE_THRESHOLD_DB)

    # Generate nonsilent chunks (start, end) with pydub
    response_timing_chunks = np.array(
        silence.detect_nonsilent(normalized_sound, min_silence_len=MIN_PERIOD_SILENCE_MS,
                               silence_thresh=SILENCE_THRESHOLD_DB,
                               seek_step=1))

    # If unable to detect nonsilence, end program and notify user
    if len(response_timing_chunks) == 0:
        print("Could not detect user's responses. Silence threshold/Minimum silence period may need tuning.")
        exit(1)

    # Calculate the time that the user starts to speak in each nonsilent "chunk"
    response_timing_markers = np.array(response_timing_chunks[:, 0]) / 1000.0
    while response_timing_markers[0] == 0.0:
        response_timing_markers = np.delete(response_timing_markers, 0)
        response_timing_chunks = np.delete(response_timing_chunks, 0, 0)

    # Create a folder to store the individual responses as clips to help determine
    # response accuracies later on.
    clip_seperation_path = TRIAL_NAME + "_reponse_chunks"
    if not os.path.isdir(clip_seperation_path):
        os.mkdir(clip_seperation_path)
    # How much we add (ms) to the ends of a clip when saved
    clip_threshold = 600
    for i in range(len(response_timing_chunks)):
        chunk = response_timing_chunks[i]
        chunk_filename = os.path.join(clip_seperation_path, f"chunk{i}.wav")
        # Save the chunk as a serperate wav, acounting for the fact it could be at the very beggining or end
        if chunk[0] <= clip_threshold:
            (audio_segment[0:chunk[1] + clip_threshold]).export(chunk_filename, format="wav")
        elif chunk[1] >= ((rec_seconds * 1000.0) - clip_threshold - 1):
            (audio_segment[chunk[0] - clip_threshold:(rec_seconds * 1000) - 1]).export(chunk_filename, format="wav")
        else:
            (audio_segment[chunk[0] - clip_threshold:chunk[1] + clip_threshold]).export(chunk_filename,
                                                                                        format="wav")
        # Reformat the wav files using soundfile to allow for speech recongition, and store in folder
        data, samplerate = soundfile.read(chunk_filename)
        soundfile.write(chunk_filename, data, samplerate, subtype='PCM_16')

    # Create an array to hold raw responses of user
    raw_responses = []
    # Create an array to hold users response accuracy (TRUE, FALSE, or N/A)
    response_accuracies = []

    # Init the speech to text recognizer
    r = sr.Recognizer()

    # Calculate the reponse times given the arrays for response_timing_markers and stimuli_time_stamps
    reaction_times = []
    num_correct_responses = 0
    for i in range(NUM_TESTS):
        # If there is no response after a time stamp, clearly the user failed to respond...
        if stimuli_time_stamps[i] > response_timing_markers[-1]:
            rt = float('nan')
            response_accuracies.append("N/A")
            raw_responses.append("N/A")
        else:
            # Determine the most accurate nonsilent chunk that is associated with a given iteration
            for j in range(len(response_timing_markers)):
                if response_timing_markers[j] > stimuli_time_stamps[i]:
                    # If reaction is too fast, it means the program is considering a delayed response from previous stimulus
                    # Thus, we should continue the loop if that is the case, otherwise, break and store the reaction time
                    if response_timing_markers[j] - stimuli_time_stamps[i] < 0.1 and len(reaction_times) > 0 and reaction_times[-1] > AFTER_MIN_DELAY:
                        continue
                    rt = response_timing_markers[j] - stimuli_time_stamps[i]
                    break
            # If there is no nonsilent chunk after the time that the stimulus is displayed, store reaction time as "nan"
            # Also if the user's response is over 1.6s after the stimulus is displayed, then we know they either failed to
            # respond or the audio was not recorded and intepreted properly.
            if j >= len(response_timing_markers):
                reaction_times.append(float('nan'))
                raw_responses.append("N/A")
                response_accuracies.append("N/A")
                continue
            else:
                # If the response was valid, detemine if it was correct using speech recognition
                with sr.AudioFile(os.path.join(clip_seperation_path, f"chunk{j}.wav")) as source:
                    # listen for the data (load audio to memory)
                    audio_data = r.record(source)
                    # recognize (convert from speech to text)
                    try:
                        resp = (r.recognize_google(audio_data).split()[0]).upper()
                    # If no response can be determined, report accuracies as N/A, store reaction time, and move on
                    except sr.UnknownValueError as err:
                        response_accuracies.append("N/A")
                        raw_responses.append("N/A")
                        reaction_times.append(rt)
                        continue
                    # compare response from stt to the actual response, update response_accuracies accordingly
                    if (resp == "YES" and answer_array[i][0] == "Yes") or (resp == "NO" and answer_array[i][0] == "No"):
                        response_accuracies.append("TRUE")
                        num_correct_responses += 1
                    else:
                        response_accuracies.append("FALSE")
                    raw_responses.append(resp)
        reaction_times.append(rt)

    # Notify user of performance
    print("You got " + str(num_correct_responses) + " / " + str(NUM_TESTS) +
          " correct answers (" + str(100 * float(num_correct_responses) / NUM_TESTS) + " %).")

    # Create another array to label each reaction time according to if it was within the allotted time or not
    reaction_on_time = np.empty(NUM_TESTS, dtype=bool)
    for i in range(NUM_TESTS):
        if reaction_times[i] > AFTER_MIN_DELAY or isnan(reaction_times[i]):
            reaction_on_time[i] = False
        else:
            reaction_on_time[i] = True

    # Write results to file
    with open(TRIAL_NAME + ".csv", 'w') as reac_file:
        writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Hour', 'Minute', 'Correct answer', 'User Response', 'Accuracy (T/F)', 'Reaction time (s)', 'Reaction on time (T/F)'])
        for i in range(NUM_TESTS):
            writer.writerow([hour_array[i], minute_array[i], answer_array[i][0],
                             raw_responses[i], response_accuracies[i], reaction_times[i],
                             reaction_on_time[i]])
    print("Done")
