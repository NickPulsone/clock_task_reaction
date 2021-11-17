# clock_task_reaction
Implements psychological clock test. Usage: ./python clock_test.py
The program will audibly dictate a series of times (hour, minute), to which the user must respond to. The user, to the best of their ability, respond "yes" if the hour hand and the minute hand lie on the same (vertical) half of the clock. The user must responsd "no" if otherwise.

Requires Python 3.9. Edit tunable paramaters as necessary in "color_reaction.py."

IMPORTANT: Include the files in this drive link in your working directory (too big for github): https://drive.google.com/drive/folders/1_XCEDEXR9AgY9L-dRdYDVTmz9gXPXfcK?usp=sharing

If the program is unable to calculate the reaction time of a given response (whether it be the because the user failed to respond, the microphone did not pick up user audio, or otherwise) the reaction time will be recorded as "nan."

For actual testing, use the programs in the "post_processing" branch. "run_clock.py" will run the test on a subject and save an audio file and csv file with the relevant data. "process_clock.py" will use this data to calculate reaction times, accuracies, etc. Post processing is mostly automatic, but does require review from a user to doule check the responses.
