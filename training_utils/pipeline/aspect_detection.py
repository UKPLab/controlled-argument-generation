# Original work Copyright 2020, Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt

import json
import requests
import os
from tqdm import tqdm
import argparse
import nltk
from datetime import datetime
from credentials import userID, APIkey

"""
Reads the arguments classified by argument_classification.py and adds aspect information to each sample.
All samples are merged into a single file.
"""

def identify_aspects(topic, payload_data):
    payload = {
        "query": topic,
        "arguments": payload_data,
        "userID": userID,
        "apiKey": APIkey
    }

    is_timed_out = True
    timed_out_ctr = 1
    while is_timed_out == True:
        try:
            json_dict = requests.post("https://api.argumentsearch.com/en/get_aspects", timeout=300, data=json.dumps(payload),
                                      headers={'Content-Type': 'application/json'}).json()
            is_timed_out = False
        except requests.exceptions.Timeout or ConnectionError or json.decoder.JSONDecodeError:
            print("Timed out for {0} times.".format(str(timed_out_ctr)))
        except Exception as e:
            print(e)

    return json_dict


def append_to_file(full_data, result_json_dict, out_f):
    ordered_ids = sorted([int(k) for k in result_json_dict["arguments"].keys()])

    for id in ordered_ids:
        full_data[str(id)]["aspect_string"] = result_json_dict["arguments"][str(id)]["aspects"]
        full_data[str(id)]["aspect_pos"] = result_json_dict["arguments"][str(id)]["aspects_pos"]
        full_data[str(id)]["sent"] = result_json_dict["arguments"][str(id)]["sent"]
        full_data[str(id)]["id"] = id

        if id > 0:
            out_f.write("\n")
        json.dump(full_data[str(id)], out_f)

def find_new_doc_id(path):
    if not os.path.isfile(path+"../aspect_detection_starting_doc_id.txt"):
        return 0
    with open(path+"../aspect_detection_starting_doc_id.txt") as in_f:
        new_id = list(in_f.readlines())
        if len(new_id) > 0:
            return int(new_id[0])
        else:
            return 0

def get_total_id(path, out_file):
    if not os.path.isfile(path + out_file):
        return 0, set()
    with open(path + out_file, "r") as out_f:
        latest_id = -1
        duplicates = set()
        for line in out_f.readlines():
            #json_dict = json.loads(line)
            json_dict = json.loads(line)
            duplicates.add(json_dict["sent"])
            latest_id = json_dict["id"]
        return latest_id+1, duplicates

def classify_and_write(topic, doc_id_start, internal_doc_sample_id, path, out_file):
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass

    try:
        print("Start classifying aspects")
        docs = sorted([f for f in os.listdir(path) if f.startswith("final")])
        num_duplicates = 0
        pbar = tqdm()
        skipped_samples = 0 # samples skipped by aspect model
        total_samples_seen = 0

        if doc_id_start == -1: # get doc id to start from
            doc_id_start = find_new_doc_id(path)

        total_id, duplicates = get_total_id(path, out_file) # get total sample id to start from (e.g after crash)

        initial_loop = True # we may have to start from a specific internal_doc_sample_id, but only in the first loop
        with open(path + out_file, "a") as out_f:

            for doc_id in tqdm(range(doc_id_start, len(docs))):

                doc = docs[doc_id]

                if initial_loop == False:
                    print("Doc number {0} - resetting internal_doc_sample_id".format(str(doc_id)))
                    internal_doc_sample_id = 0

                with open(path+doc, "r") as in_f:

                    full_data = {}
                    payload_data = {}
                    for line_number, line in enumerate(in_f):
                        if initial_loop == True and line_number < internal_doc_sample_id:
                            continue # skip until the first sample after the crash

                        try:
                            row = json.loads(line)
                        except json.decoder.JSONDecodeError as jde:
                            print(jde)
                            print("Skipping sample at doc_id {0}, total_sample_id {1} and internal_doc_sample_id {2} "
                                  "due two several JSON dicts in one line".format(
                                str(doc_id),
                                str(total_id),
                                str(internal_doc_sample_id)))
                            internal_doc_sample_id += 1
                            total_samples_seen += 1
                            skipped_samples += 1
                            continue

                        total_samples_seen += 1
                        internal_doc_sample_id += 1

                        # check duplicates
                        duplicate_check = row["sent"].replace("\xad\xad", "-").replace("\u2026", " ...") \
                            .replace("\n", " ").replace("\r", " ").replace("...", " ...")
                        duplicate_check = " ".join(nltk.word_tokenize(duplicate_check))
                        if duplicate_check in duplicates:
                            num_duplicates += 1
                            continue

                        # add rows to dict
                        full_data[str(total_id)] = row
                        payload_data[str(total_id)] = row["sent"]
                        duplicates.add(duplicate_check)

                        # if the dict has reached a certain limit, identify aspects and write to file
                        if total_id > 0 and total_id % 200 == 0:
                            result_json_dict = identify_aspects(topic, payload_data)
                            skipped_samples += (len(payload_data)-len(result_json_dict["arguments"]))
                            append_to_file(full_data, result_json_dict, out_f)
                            full_data = {}
                            payload_data = {}

                        total_id += 1
                        pbar.update(1)

                        if total_id > 0 and total_id % 50000 == 0:
                            print("\nDuplicates removed: {0}".format(str(num_duplicates)))
                            print("Samples skipped by aspect model: {0}".format(str(skipped_samples)))
                            print("Actual samples in file: {0}".format(str(total_id-1-skipped_samples)))
                            print("Total samples, including skipped: {0}".format(str(total_id-1)))
                            print("Total samples seen: {0}\n".format(str(total_samples_seen-1)))

                    if len(full_data) > 0: # last execution of leftover (< 200) samples
                        result_json_dict = identify_aspects(topic, payload_data)
                        skipped_samples += (len(payload_data) - len(result_json_dict["arguments"]))
                        append_to_file(full_data, result_json_dict, out_f)

                initial_loop = False

            print("\nProcess finished!")

            print_out = "\nDuplicates removed: {0}".format(str(num_duplicates))
            print_out += "\nSamples skipped by aspect model: {0}".format(str(skipped_samples))
            print_out += "\nActual samples in file: {0}".format(str(total_id - 1 - skipped_samples))
            print_out += "\nTotal samples, including skipped: {0}".format(str(total_id - 1))
            print_out += "\nTotal samples seen: {0}\n".format(str(total_samples_seen - 1))

            print(print_out)

            with open(path + "../note.txt", "a") as note_out_f:
                note_out_f.write("\n\n" + str(datetime.now()) + " - aspect_classification.py:\n")
                note_out_f.write(print_out)

    except Exception as e:
        print(e)
        print("Crashed at doc_id {0} and internal_doc_sample_id {1}".format(str(doc_id), str(internal_doc_sample_id)))
        print("For restart, start at given doc_id and internal_sample_id+1.")

parser = argparse.ArgumentParser(description='Classifies aspects from docs with argument samples')
parser.add_argument('--topic', type=str, required=True,
                                        help='Topic without underscore')
parser.add_argument('--doc_id_start', type=int, default=-1,
                                        help='The doc id from which to start after crash/timeout. Leave at -1 if no crash occured.')
parser.add_argument('--internal_doc_sample_id', type=int, default=0,
                                        help='Line number within doc to start from.')
parser.add_argument('--index', type=str, default="common-crawl-en", required=True,
                                        help='Source index')

args = parser.parse_args()
topic = args.topic

training_data_path = "../../training_data/"

# doc id to start from: this is important in case of a crash or if new arguments have been classified and aspects are to be added
# In case of the latter, a file with the latest doc id to start from is read (this is created by the argument_classification.py)
doc_id_start = args.doc_id_start

internal_doc_sample_id = args.internal_doc_sample_id # line number within doc to start from in case of crash
input_path = "{0}{1}/{2}/processed/".format(training_data_path, args.index, topic.replace(" ", "_")) # path to folder with argument docs
classify_and_write(topic, doc_id_start, internal_doc_sample_id, input_path, "merged.jsonl")
