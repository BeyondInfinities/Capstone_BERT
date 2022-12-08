#!/bin/bash
# A simple script to convert the Wordnet database into files containing
# adjectives, adverbs, noin, sense, verb. For use with BERT Bias

# Download the Wordnet database
wget http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz

# Extract the database
tar -xvzf WordNet-3.0.tar.gz

cd dict

# Convert the database into a format that can be used by BERT Bias
egrep -o "^[0-9]{8}\s[0-9]{2}\s[a-z]\s[0-9]{2}\s[a-zA-Z_]*\s" data.adj | cut -d ' ' -f 5 > adj.txt
egrep -o "^[0-9]{8}\s[0-9]{2}\s[a-z]\s[0-9]{2}\s[a-zA-Z_]*\s" data.adv | cut -d ' ' -f 5 > adv.txt
egrep -o "^[0-9]{8}\s[0-9]{2}\s[a-z]\s[0-9]{2}\s[a-zA-Z_]*\s" data.noun | cut -d ' ' -f 5 > noun.txt
egrep -o "^[0-9]{8}\s[0-9]{2}\s[a-z]\s[0-9]{2}\s[a-zA-Z_]*\s" data.verb | cut -d ' ' -f 5 > verb.txt

