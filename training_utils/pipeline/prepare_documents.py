# Original work Copyright 2020, Ubiquitous Knowledge Processing (UKP) Lab, Technische Universität Darmstadt

from collections import defaultdict
import json
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from collections import Counter
from copy import deepcopy
import argparse
from datetime import datetime
import sys
import os

"""
Takes output file from aspect classification as input and prepares documents by concatenating arguments of the 
same topic, stance, and aspect.
"""

stopwords = set(stopwords.words('english'))

def create_dirs(paths):
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def stem_aspect(stemmer, aspect_string):
    stemmed = ""
    for aspect in aspect_string.split(" "):
        stemmed = stemmed + stemmer.stem(aspect) + " "
    return stemmed.rstrip()

def simplify_aspect(aspect):
    return " ".join([t for t in aspect.split(" ") if t not in stopwords])

def preprocess_aspects(aspect_list):
    new_aspect_list = []
    for aspect in aspect_list:
        aspect = aspect.lower()

        if aspect in new_aspect_list:
            continue

        split_aspect = aspect.split(" ")

        if split_aspect[-1] in stopwords:
            del split_aspect[-1]

        if len(split_aspect) > 0 and split_aspect[0] in stopwords:
            del split_aspect[0]

        new_aspect_list.append(" ".join(split_aspect))

    return new_aspect_list

def parse_aspects(file, data_path, final_data, topic):
    stemmer = nltk.stem.SnowballStemmer("english")
    aspect_clusters_stem = defaultdict(list)
    score_skipped = 0
    duplicate_ctr = 0
    with open(data_path + file, "r") as in_f:
        ctr = 0
        all_aspects = set()
        for line in tqdm(in_f.readlines()):
            if line == "":
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                print(e)

            stance = data["stance"]
            aspect_list = data["aspect_string"]
            argument = data["sent"]

            aspect_list = preprocess_aspects(aspect_list) # removes stopwords from aspects (beginning and end)

            try:
                if not topic in final_data.keys():
                    final_data[topic] = {}

                if not stance in final_data[topic].keys():
                    final_data[topic][stance] = {}

                for aspect in aspect_list:
                    stemmed = stem_aspect(stemmer, aspect)
                    stemmed = simplify_aspect(stemmed)
                    aspect_clusters_stem[stemmed].append(aspect)

                    if not stemmed in final_data[topic][stance].keys():
                        final_data[topic][stance][stemmed] = []

                    if argument in final_data[topic][stance][stemmed]:
                        duplicate_ctr += 1 #  same aspect with diff. capitalization appeared before or aspect is several times in the sentence
                    else:
                        final_data[topic][stance][stemmed].append((argument, data["doc_score"]))
                        all_aspects.add(aspect)
                        ctr += 1
            except Exception as e:
                print(e)

    print("Total unique aspects: {0}, total aggregated aspects: {1}".format(str(len(all_aspects)), str(len(aspect_clusters_stem))))
    print("Finished with {0} sentences (some sents have multiple aspects and they are clustered several times).".format(str(ctr)))
    print("Skipped {0} sentences due to low document score.".format(str(score_skipped)))
    print("Skipped {0} sentences due to duplicates.".format(str(duplicate_ctr)))
    return final_data, aspect_clusters_stem, all_aspects

def create_training_docs(clustered_aspect_data, output_path, note_path, generation_path, MAX_SENTS,
                         exclude_aspects, use_stance="PRO"):

    total_sents = 0

    total_clusters_before = 0
    total_clusters_after = 0

    lowest_cluster_size_found = sys.maxsize

    removed_aspects = []

    top_aspects = []
    top_aspects_max_size = []

    first_aspect_generated = True
    for topic, values in clustered_aspect_data.items():
        for stance, aspects in values.items():
            total_clusters_before += len(aspects)
            print("Processing stance " + stance + " of topic " + topic)

            temp_stance = "PRO" if stance == "Argument_for" else "CON"

            if temp_stance != use_stance:
                continue

            length_sorted_keys = sorted(aspects, key=lambda k: len(aspects[k]), reverse=True)

            for aspect_loop_id, aspect_name in enumerate(length_sorted_keys):
                aspect_name = aspect_name.lower()
                aspect_args = aspects[aspect_name]

                if aspect_name in exclude_aspects:
                    continue

                exclude_aspects.add(aspect_name)

                if total_sents >= MAX_SENTS:
                    continue

                if len(aspect_args) < MIN_ASPECT_CLUSTER_SIZE:
                    continue

                if len(aspect_name) < 3 or aspect_name in stopwords or len([False for s in aspect_name.split(" ") if s in stopwords]) == len(aspect_name.split(" ")):
                    continue

                if os.path.isfile(generation_path+"control_codes.jsonl"):
                    first_aspect_generated = False

                with open(output_path + topic.replace(" ", "_") + "_" + temp_stance + "_" + aspect_name.replace(" ", "_").replace("/", "_") + ".txt",
                          "a") as out_f, open(generation_path+"control_codes.jsonl", "a") as generation_out:

                    if len(aspect_args) < lowest_cluster_size_found:
                        lowest_cluster_size_found = len(aspect_args)

                    total_clusters_after += 1
                    first_aspect_arg = True
                    text_hashes = set()
                    total_cluster_sents = 0
                    top_aspects.append((aspect_name, len(aspect_args)))

                    aspect_args_sorted = sorted(aspect_args, key=lambda k: k[1], reverse=True)
                    args_count = 0
                    for args_count, (argument, _) in enumerate(aspect_args_sorted):
                        if hash(argument) not in text_hashes:

                            if total_cluster_sents >= MAX_ASPECT_CLUSTER_SIZE:
                                continue

                            if total_sents >= MAX_SENTS and total_cluster_sents >= MIN_ASPECT_CLUSTER_SIZE:
                                continue
                            text_hashes.add(hash(argument))
                            total_sents += 1
                            total_cluster_sents += 1
                            if first_aspect_arg == True:
                                out_f.write(argument.replace("\n", " ").replace("\r", " "))
                                first_aspect_arg = False
                            else:
                                out_f.write(" " + argument.replace("\n", " ").replace("\r", " "))
                    else:
                        top_aspects_max_size.append((aspect_name, total_cluster_sents))
                        if args_count > 0:
                            generation_out.write(("\n" if first_aspect_generated == False else "") +
                                             json.dumps({"topic": topic, "stance": temp_stance, "aspect": aspect_name.replace("/", " ")}))
                            first_aspect_generated = False

    printout = "\n==========================\n"
    printout += "Stats for stance " + use_stance +"\n"
    printout += "==========================\n"
    printout += "Total sents added: "+str(total_sents) + "\n"
    printout += "Total cluster before processing: "+str(total_clusters_before) + "\n"
    printout += "Total cluster after processing: "+str(total_clusters_after) + "\n"
    printout += "Top aspects with number of arguments (before MAX_ASPECT_CLUSTER_SIZE): "+str(sorted(top_aspects, key=lambda k: k[1], reverse=True)[:5]) + "\n"
    printout += "Top aspects with number of arguments (after MAX_ASPECT_CLUSTER_SIZE): "+str(sorted(top_aspects_max_size, key=lambda k: k[1], reverse=True)[:5]) + "\n"
    printout += "Removed aspects that are contained in topic name: " + str(removed_aspects) + "\n"
    printout += "Smallest cluster size: "+str(lowest_cluster_size_found) + "\n"
    print(printout)

    print("Top aspects with number of arguments: " + str(sorted(top_aspects, key=lambda k: k[1], reverse=True)) + "\n")

    with open(note_path + "note.txt", "a") as out_f:
        out_f.write("\n" + printout)

    return exclude_aspects, total_sents, top_aspects_max_size

def update_aggregated_aspect_labels(topic_dict, aspect_clusters_lemmatize):
    new_dict = {}
    for topic, values in topic_dict.items():
        if topic not in new_dict.keys():
            new_dict[topic] = {}
        for stance, aspects in values.items():
            if stance not in new_dict[topic].keys():
                new_dict[topic][stance] = {}
            for aspect_name, aspect_sentences in tqdm(aspects.items()):
                new_name = Counter(aspect_clusters_lemmatize[aspect_name]).most_common(1)[0][0]
                new_dict[topic][stance][new_name] = deepcopy(topic_dict[topic][stance][aspect_name])
    del topic_dict
    return new_dict

def store_preprocessed(preprocessed_clusters_path, topic_dict, aspect_clusters_stem, all_aspects):
    with open(preprocessed_clusters_path+"topic_dict.json", "w") as out_f:
        json.dump(topic_dict, out_f, indent=4)
    with open(preprocessed_clusters_path+"aspect_clusters_stem.json", "w") as out_f:
        json.dump(aspect_clusters_stem, out_f, indent=4)
    with open(preprocessed_clusters_path+"all_aspects.txt", "w") as out_f:
        for i, a in enumerate(all_aspects):
            if a != "":
                out_f.write(a)
                if i < len(all_aspects)-1:
                    out_f.write("\n")

def load_processed(preprocessed_clusters_path):
    with open(preprocessed_clusters_path+"topic_dict.json", "r") as in_f:
        topic_dict = json.load(in_f)
    with open(preprocessed_clusters_path+"aspect_clusters_stem.json", "r") as in_f:
        aspect_clusters_stem = json.load(in_f)
    with open(preprocessed_clusters_path+"all_aspects.txt", "r") as in_f:
        all_aspects = set()
        for line in in_f.readlines():
            if line != "":
                all_aspects.add(line)
    return topic_dict, aspect_clusters_stem, all_aspects

def check_for_stored_files(preprocessed_clusters_path):
    if os.path.isfile(preprocessed_clusters_path+"topic_dict.json") and os.path.isfile(preprocessed_clusters_path+"aspect_clusters_stem.json") and\
        os.path.isfile(preprocessed_clusters_path+"all_aspects.txt"):
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Creates final training documents')
parser.add_argument('--topic', type=str, required=True,
                                        help='Topic without underscore')
parser.add_argument('--min_aspect_cluster_size', type=int, default=15,
                                        help='Each training document for an aspect has to hold at least N arguments.')
parser.add_argument('--use_stemmer', type=int, default=1,
                                        help='If aspects should be stemmed before clustering.')
parser.add_argument('--max_sents', type=int, default=200000,
                                        help='Maximum number of sents to use for training.')
parser.add_argument('--index', type=str, default="common-crawl-en", required=True,
                                        help='Source index')
parser.add_argument('--max_aspect_cluster_size', type=int, default=1500,
                                        help='Each training document for an aspect may hold at max. N arguments.')
parser.add_argument('--store_aspect_clusters', type=int, default=0,
                                        help='Stores aspect clusters for faster recalculation of training data. (Attention: Files might get very large) .')

args = parser.parse_args()

# parse args
topic = args.topic
store_aspect_clusters = args.store_aspect_clusters
MIN_ASPECT_CLUSTER_SIZE = args.min_aspect_cluster_size # each aspect has to hold at least N arguments
MAX_ASPECT_CLUSTER_SIZE = args.max_aspect_cluster_size # each aspect has to hold at least N arguments
USE_STEMMER = True if args.use_stemmer == 1 else False # cluster arguments by stemmed aspects instead of original aspect
MAX_SENTS = args.max_sents

# paths
training_data_path = "../../training_data/"
input_path = "{0}{1}/{2}/processed/".format(training_data_path, args.index, topic.replace(" ", "_")) # JSONL document with arguments and their aspects
output_path = "{0}{1}/{2}/final/".format(training_data_path, args.index, topic.replace(" ", "_")) # Final training document files
generation_path = "{0}{1}/{2}/control_codes/".format(training_data_path, args.index, topic.replace(" ", "_")) # Files with used control codes for later generation
preprocessed_clusters_path = "{0}{1}/{2}/tmp_cluster_files/".format(training_data_path, args.index, topic.replace(" ", "_")) # temp path for preprocessedaspect clusters
input_file = "merged.jsonl"

create_dirs([output_path, generation_path, preprocessed_clusters_path])

begin_text = 'Start preparing documents with topic "{0}", MIN_ASPECT_CLUSTER_SIZE "{1}"'\
      ', USE_STEMMER "{2}", MAX_SENTS "{3}" and MAX_ASPECT_CLUSTER_SIZE "{4}"'.format(str(topic),
                                                      str(MIN_ASPECT_CLUSTER_SIZE), str(USE_STEMMER), str(MAX_SENTS), str(MAX_ASPECT_CLUSTER_SIZE))
print(begin_text)
with open(input_path + "../note.txt", "a") as out_f:
    out_f.write("\n\n" + str(datetime.now()) + " - pepare_documents.py:\n")
    out_f.write(begin_text)

if check_for_stored_files(preprocessed_clusters_path): # load aspect clusters if exist
    print("Found preprocessing files on disk...loading...")
    clustered_aspect_data, stemmed_aspect_lookup, all_aspects = load_processed(preprocessed_clusters_path)
else: # create clusters of aspect with all belonging arguments, labelled by the stemmed aspects
    print("No preprocessing files founding on disk...start creating...")
    clustered_aspect_data, stemmed_aspect_lookup, all_aspects = parse_aspects(input_file, input_path, {}, topic)
    if store_aspect_clusters == 1:
        store_preprocessed(preprocessed_clusters_path, clustered_aspect_data, stemmed_aspect_lookup, all_aspects)

# As we dont want stemmed aspects as control code, replace the stemmed aspect labels with the most common unstemmed variant
# (as seen in all arguments with that aspect)
if USE_STEMMER == True:
    clustered_aspect_data = update_aggregated_aspect_labels(clustered_aspect_data, stemmed_aspect_lookup)


# creates the training documents that hold arguments with the same topic/stance/aspect
aspects_pro_used, total_pro_sents, all_aspects_PRO = create_training_docs(clustered_aspect_data, output_path, input_path + "../", generation_path,
                                                                          int(MAX_SENTS/2), set(), use_stance="PRO")
aspects_con_used, total_con_sents, all_aspects_CON = create_training_docs(clustered_aspect_data, output_path, input_path + "../", generation_path,
                                                                          int(MAX_SENTS/2) + ((MAX_SENTS/2)-total_pro_sents),
                                                                          set(), use_stance="CON")

# in case there were not enough con arguments in the training data, try to fill up with pro arguments (simplify this process)
if MAX_SENTS - (total_con_sents + total_pro_sents) > MIN_ASPECT_CLUSTER_SIZE:
    aspects_pro_used, total_pro_sents, all_aspects_PRO_temp = create_training_docs(clustered_aspect_data, output_path, input_path, generation_path,
                                                                                                   MAX_SENTS - (total_con_sents + total_pro_sents),
                                                                                                   aspects_pro_used, use_stance="PRO")