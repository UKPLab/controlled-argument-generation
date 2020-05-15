#!/usr/bin/env bash

MIN_ASPECT_CLUSTER_SIZE=15 # each training document must hold at least MIN_ASPECT_CLUSTER_SIZE arguments
MAX_ASPECT_CLUSTER_SIZE=1500  # each training document must hold at most MAX_ASPECT_CLUSTER_SIZE arguments
MAX_SENTS=200000 # max number of training sentences for each topic (will be balanced upon PRO and CON arguments)
STORE_ASPECT_CLUSTERS=0 # stores aspect clusters to [topic_folder]/tmp_cluster_files to save time for future training document generation
INDEX="common-crawl-en" # source for training data, either redditcomment-en or common-crawl-en
TOPICS=("marijuana legalization" "nuclear energy" "school uniforms" "death penalty" "minimum wage" "gun control" "cloning" "abortion")

for run_topic in "${TOPICS[@]}" ; do
    python prepare_documents.py --max_sents $MAX_SENTS --topic "${run_topic}" --index $INDEX --store_aspect_clusters $STORE_ASPECT_CLUSTERS --max_aspect_cluster_size $MAX_ASPECT_CLUSTER_SIZE --min_aspect_cluster_size $MIN_ASPECT_CLUSTER_SIZE
done

