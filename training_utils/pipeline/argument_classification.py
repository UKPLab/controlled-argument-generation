# Original work Copyright 2020, Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt

import json
import requests
import os
from tqdm import tqdm
import argparse
from datetime import datetime
from credentials import userID, APIkey

"""
Reads documents with split sentences, classifies them into pro-/con-/non-arguments, and writes out arguments and their stance information.
"""

def read_doc(path):
    with open(path, "r") as in_f:
        return json.load(in_f)

def get_doc_list(path):
    if os.path.isfile(path+"../processed_docs.txt"):
        with open(path+"../processed_docs.txt", "r") as in_f:
            processed_docs = in_f.readlines()
            if len(processed_docs) == 0:
                return sorted(os.listdir(path))
            processed_docs = processed_docs[0].split(",")
            if processed_docs[-1] == "":
                processed_docs = processed_docs[:-1]
            processed_docs = set(processed_docs)
            all_docs = set(sorted(os.listdir(path)))
            return sorted(all_docs.difference(processed_docs))
    else:
        return sorted(os.listdir(path))

def get_out_number(doc_path):
    """
    If new documents are gathered after previous documents were already processed via aspect detection, we start a new final.jsonl file
    and store the ID so the aspect_classification.py process knows where to start from.
    """
    latest_file_ids = sorted([int(f.replace("final-", "").replace(".jsonl", "")) for f in os.listdir(doc_path) if f.startswith("final-")])
    if len(latest_file_ids) == 0:
        return 0
    new_id = latest_file_ids[-1] + 1
    with open(doc_path+"../aspect_detection_starting_doc_id.txt", "w") as out_f:
        out_f.write(str(new_id))
    return new_id


def write_data(topic, doc_id_start, topic_word_list, MAX_FILE_SIZE, FILTER_TOPIC, out_path):
    print('Start classifying sentences for topic "{0}" from doc_id_start {1} '
          'with MAX_FILE_SIZE {2}, and FILTER_TOPIC set "{3}"". Writing to {4}'.format(topic, str(doc_id_start),
                                                                                   str(MAX_FILE_SIZE),
                                                                                   str(FILTER_TOPIC), str(out_path)))

    try:
        os.makedirs(out_path + "/processed/")
    except FileExistsError:
        # directory already exists
        pass

    try:
        doc_path = out_path + "/unprocessed/"
        out_path = out_path + "/processed/"
        docs = get_doc_list(doc_path)
        first_line = True
        total_id = 0
        total_processed = 0
        out_number = get_out_number(out_path) # each time this method is called, it starts a new document
        current_file_size = 0
        sents_skipped_relevance = 0
        total_duplicates = 0
        for doc_id in tqdm(range(doc_id_start, len(docs))):
            doc = docs[doc_id]
            if current_file_size >= MAX_FILE_SIZE:
                out_number += 1
                first_line = True
                current_file_size = 0

            with open(doc_path+doc, "r") as in_f, open(out_path+"final-{0}.jsonl".format(out_number), "a") as out_f,\
                open(doc_path+"../processed_docs.txt", "a") as out_processed_docs_f:
                data = json.load(in_f)

                sent_ids = list(range(0,len(data['sents'])))
                if FILTER_TOPIC == True:
                    for sent_i, sent in enumerate(data['sents']):
                        total_processed += 1
                        if not True in [True for tok in topic_word_list if tok.lower() in sent.lower()]:
                            sent_ids.remove(sent_i)
                            sents_skipped_relevance += 1


                sents_to_send = [data['sents'][s] for s in sent_ids]
                if len(sents_to_send) == 0:
                    continue

                payload = {
                        "topic": topic,
                        "showOnlyArguments": False,
                        "computeAttention": False,
                        "sortBy": "none",
                        "predictStance": True,
                        "sentences": sents_to_send,
                        "userID": userID,
                        "apiKey": APIkey
                    }

                is_timed_out = True
                timed_out_ctr = 1
                while is_timed_out == True:
                    try:
                        json_dict = requests.post("https://api.argumentsearch.com/en/classify", timeout=300, data=json.dumps(payload),
                                                          headers={'Content-Type': 'application/json'}).json()
                        is_timed_out = False
                    except requests.exceptions.Timeout or ConnectionError:

                        print("Timed out for {0} times.".format(str(timed_out_ctr)))


                header = { #keys from the document json-file that will be taken over to the training data samples
                    "doc_id": data.get("id", -1),
                    "doc_metadata_id": data.get("metadata_id", "N/A"),
                    "doc_url": data.get("url", "N/A"),
                    "doc_score": data.get("score", -1),
                    "index": data.get("index", "N/A")
                }


                duplicates = set()
                for doc_sent_id, sent in zip(sent_ids, json_dict['sentences']):
                    if sent['sentenceOriginal'] in duplicates:
                        total_duplicates += 1
                        continue

                    duplicates.add(sent['sentenceOriginal'])

                    if sent["argumentLabel"] == "argument":
                        row = {"id": doc_id_start+total_id,
                               "doc_sent_id": doc_sent_id,
                               "stance": "Argument_for" if sent["stanceLabel"] == "pro" else "Argument_against",
                               "sent": sent["sentenceOriginal"]}
                        row.update(header)

                        if first_line == False:
                            out_f.write("\n")
                        json.dump(row, out_f)
                        first_line = False
                        total_id += 1
                        current_file_size += 1
                        out_processed_docs_f.write(doc+",")

        print_out = str(datetime.now()) + " - Total sentences skipped due to topic-relevance: {0}\n".format(str(sents_skipped_relevance))
        print_out += str(datetime.now()) + " - Total duplicates skipped: {0}\n".format(str(total_duplicates))
        print_out += str(datetime.now()) + " - Total sentences left: {0}\n".format(str(total_id))
        print_out += str(datetime.now()) + " - Total sentences processed: {0}\n".format(str(total_processed))
        print_out += str(datetime.now()) + " - The rest of the sentence were filtered out because they are no arguments."
        print(print_out)

        with open(out_path+"../note.txt", "a") as note_out_f:
            note_out_f.write("\n\n"+str(datetime.now()) + " - arg_classification.py:\n")
            note_out_f.write(print_out)

    except Exception as e:
        print(e)
        print("Crashed at doc_id {0}".format(str(doc_id)))

parser = argparse.ArgumentParser(description='Classifies arguments from raw docs')
parser.add_argument('--topic', type=str, required=True,
                                        help='Topic without underscore.')
parser.add_argument('--doc_id_start', type=int, default=0,
                                        help='The doc id from which to start after crash. At the beginning defaults to 0.')
parser.add_argument('--max_file_size', type=int, default=200000,
                                        help='Maximum lines of a document before splitting it.')
parser.add_argument('--filter_topic', type=int, default=1,
                                        help='Filters out sentences that do not contain a token from the topic or its defined synonyms.')
parser.add_argument('--index', type=str, default="common-crawl-en", required=True,
                                        help='Data source index name.')

args = parser.parse_args()

training_data_path = "../../training_data/"

topic = args.topic
doc_id_start = args.doc_id_start
MAX_FILE_SIZE = args.max_file_size
out_path = "{0}{1}/{2}/".format(training_data_path, args.index, topic.replace(" ", "_"))
FILTER_TOPIC = True if args.filter_topic == 1 else False

topic_word_dict = { # topic synonyms to pre-filter sentences prior to argument and stance classification
    "school uniforms": ["uniform", "college", "outfit", "dress", "suit", "jacket", "cloth"],
    "nuclear energy": ["fission", "fusion", "atomic energy", "nuclear power", "atomic power", "radioactive", "radioactivity"],
    "marijuana legalization": ["cannabis", "legalization of marijuana", "legal", "illegal", "law", "weed", "dope"],
    "cloning": ["clone", "cloned", "duplicate", "copy", "reproduct", "asexual"],
    "death penalty": ["capital punishment", "execution", "electric chair", "punishment", "punish"],
    "minimum wage": ["living wage", "base pay", "average wage", "low income"],
    "abortion": ["abort", "termination", "misbirth", "birth control"],
    "gun control": ["second amendment", "ownership", "arms reduction", "arms limitation"],
}

if topic not in topic_word_dict.keys():
    topic_word_dict[topic] = []
topic_word_dict[topic].extend(topic.split(" "))
write_data(topic, doc_id_start, topic_word_dict[topic], MAX_FILE_SIZE, FILTER_TOPIC, out_path)