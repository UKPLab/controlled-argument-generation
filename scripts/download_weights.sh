#!/bin/bash

# download reddit-comments model weights
wget -O reddit_seqlen256_v1.7z.001 "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2328/reddit_seqlen256_v1.7z.001?sequence=1&isAllowed=y"
wget -O reddit_seqlen256_v1.7z.002 "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2328/reddit_seqlen256_v1.7z.002?sequence=2&isAllowed=y"
wget -O reddit_seqlen256_v1.7z.003 "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2328/reddit_seqlen256_v1.7z.003?sequence=3&isAllowed=y"

mkdir ../reddit_seqlen256_v1
7za x reddit_seqlen256_v1.7z.001 -o../reddit_seqlen256_v1
rm reddit_seqlen256_v1.7z.001 reddit_seqlen256_v1.7z.002 reddit_seqlen256_v1.7z.003

# download common-crawl model weights
wget -O cc_seqlen256_v1.7z.001 "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2328/cc_seqlen256_v1.7z.001?sequence=6&isAllowed=y"
wget -O cc_seqlen256_v1.7z.002 "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2328/cc_seqlen256_v1.7z.002?sequence=5&isAllowed=y"
wget -O cc_seqlen256_v1.7z.003 "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2328/cc_seqlen256_v1.7z.003?sequence=4&isAllowed=y"

mkdir ../cc_seqlen256_v1
7za x cc_seqlen256_v1.7z.001 -o../cc_seqlen256_v1
rm cc_seqlen256_v1.7z.001 cc_seqlen256_v1.7z.002 cc_seqlen256_v1.7z.003