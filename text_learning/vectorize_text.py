#!/usr/bin/python3

import os
import joblib
import sys
from time import time

sys.path.append(os.path.abspath("C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\tools"))
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

from_sara = open(
    "C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\text_learning\\from_sara.txt", "r")
from_chris = open(
    "C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\text_learning\\from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

removal_words = {"germani", "shackleton", "chris", "sara", "sshacklensf", "cgermannsf"}  # Use a set for O(1) lookups
t0 = time()
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        base_dir = "C:\\Users\\User\\DataspellProjects\\Udacity-ML-Projexts\\ud120-projects\\tools"
        # temp_counter += 1
        if temp_counter < 200:
            path = path.strip()  # Remove any trailing newline or spaces
            full_path = os.path.join(base_dir, path.replace("/", os.sep))
            full_path_with_underscore = full_path[:-1] + "_"
            print(f"Processing: {full_path_with_underscore}")
            try:
                with open(full_path_with_underscore, "r") as email:
                    ### use parseOutText to extract the text from the opened email
                    parsed = parseOutText(email)

                    ### remove signature words using str.replace()
                    for word in removal_words:
                        parsed = parsed.replace(word, '')

                    ### append the cleaned text to word_data
                    word_data.append(parsed)

                    ### append the corresponding label to from_data
                    from_data.append(0 if name == "sara" else 1)
            except FileNotFoundError:
                print(f"File not found: {full_path_with_underscore}")

# print(temp_counter)
print("Emails Processed")
from_sara.close()
from_chris.close()
# print(word_data[152])
print(f'Time {round((time() - t0), 3)}s')

joblib.dump(word_data, open("your_word_data.pkl", "wb"))
joblib.dump(from_data, open("your_email_authors.pkl", "wb"))

### in Part 4, do TfIdf vectorization here

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(word_data)
y = vectorizer.get_feature_names_out()
print(len(y))
print(x.shape)
print(y[34597])
