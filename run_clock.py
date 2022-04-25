#!/usr/bin/env python
import numpy as np
from scipy.io import loadmat, wavfile
from time import sleep, time
import sounddevice as sd
import datetime
import csv
import pyaudio

""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  TUNABLE PARAMETERS    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
# Trial name (subject name, etc)
TRIAL_NAME = "clock_test"
# Name of the test sequence file
TEST_QUESTION_FILENAME = "clock_versionA.mat"
# Number of tests
# NUM_TESTS = 90
NUM_TESTS = 20
# Delay after the minute (not exact due to inconsistent timing when playing sound in python)
AFTER_HOUR_DELAY = 1.0
AFTER_MIN_DELAY = 2.0
""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

if __name__ == "__main__":
    # Load frequency data
    Number60 = loadmat("Number_60.mat")

    """
    # Initialize engine for TTS
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    """

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
    rec_seconds = int(NUM_TESTS) * 4.85
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
        # Display stimulus to screen
        print(f"Stimulus {i + 1}: \"{hour_array[i]}:{minute_array[i]},\" Correct Answer: {answer_array[i]}")
        # Play the hour sound, record time
        hour_sound = (Number60["y" + str(hour_array[i])])[:, 0]
        stream.write(hour_sound.astype(np.float32).tobytes())
        htime = time()
        # Pause, record time, then play the minute sound
        minute_sound = Number60["y" + str(minute_array[i])]
        while (time() - htime) < AFTER_HOUR_DELAY:
            sleep(0.01)
        stimuli_time_stamps[i] = datetime.datetime.now()
        stream.write(minute_sound.astype(np.float32).tobytes())
        # Pause again between stimuli
        mtime = time()
        while (time() - mtime) < AFTER_MIN_DELAY:
            sleep(0.01)
    # Close audio data stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Stop the recording, save file as .wav
    print("Waiting for recording to stop...")
    sd.wait()
    wavfile.write(TRIAL_NAME + '.wav', rec_sample_rate, myrecording)
    print("Done. Saving data...")

    # Calculate the time of each stimulus with respect to the start of the recording
    stimuli_time_stamps = np.array(
        [(stimuli_time_stamps[i] - recording_start_time).total_seconds() for i in range(NUM_TESTS)])

    # Write results to file
    with open(TRIAL_NAME + ".csv", 'w') as reac_file:
        writer = csv.writer(reac_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Hour', 'Minute', 'Correct answer', 'Stimuli time from start (s)'])
        for i in range(NUM_TESTS):
            writer.writerow([hour_array[i], minute_array[i], answer_array[i][0],
                             stimuli_time_stamps[i]])
    print("Done")
