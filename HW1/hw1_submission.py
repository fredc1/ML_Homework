import string
import re
import math
#Part ii.a cleaning the data

punctuations_to_keep = '.,?!'
start_token = '<START>'
end_token = '<END>'
human_file = 'hum.txt'
gpt_file = 'gpt.txt'

def clean_text(text):
    text = re.sub(r'[^\w\s.,?!]', '', text)
    text = text.lower()
    text = f'{start_token} {text[0:-1]} {end_token}'
    
    return text

# Read the human data file
with open(human_file, 'r') as f:
    human_data = f.readlines()
human_data = [clean_text(line) for line in human_data]

# Read the GPT data file
with open(gpt_file, 'r') as f:
    gpt_data = f.readlines()
gpt_data = [clean_text(line) for line in gpt_data]



human_partition_index = int(len(human_data) * 0.9)
human_train_data = human_data[:human_partition_index]
human_test_data = human_data[human_partition_index:]

gpt_partition_index = int(len(gpt_data) * 0.9)
gpt_train_data = gpt_data[:gpt_partition_index]
gpt_test_data = gpt_data[gpt_partition_index:]



def get_tokens(datalist):
    unique_tokens = set()
    token_pairs = set()
    token_triplets = set()
    for data in datalist:
        for line in data:
            tokens = line.split()
            unique_tokens.update(tokens)

            # Loop through each pair of adjacent tokens
            for i in range(len(tokens)-1):
                token_pairs.add((tokens[i], tokens[i+1]))

            # Loop through each triplet of adjacent tokens
            for i in range(len(tokens)-2):
                token_triplets.add((tokens[i], tokens[i+1], tokens[i+2]))
    return unique_tokens, token_pairs, token_triplets

#Part iib Train the models by counting the frequencies and get OOV rates

# Count the tokens in the training set
unique_tokens, token_pairs, token_triplets = get_tokens([human_data, gpt_data])
human_unique_dict = {}
gpt_unique_dict = {}
human_pair_dict = {}
gpt_pair_dict = {}
human_triplet_dict = {}
gpt_triplet_dict = {}

for line in human_train_data:
    tokens = line.split()

    for token in tokens:
        human_unique_dict[token] = human_unique_dict.get(token, 0) + 1

    # Loop through each pair of adjacent tokens
    for i in range(len(tokens)-1):
        pair = (tokens[i], tokens[i+1])
        human_pair_dict[pair] = human_pair_dict.get(pair, 0) + 1

    # Loop through each triplet of adjacent tokens
    for i in range(len(tokens)-2):
        triplet = (tokens[i], tokens[i+1], tokens[i+2])
        human_triplet_dict[triplet] = human_triplet_dict.get(triplet, 0) + 1

# Loop through each line in the GPT training data
for line in gpt_train_data:
    tokens = line.split()

    for token in tokens:
        gpt_unique_dict[token] = gpt_unique_dict.get(token, 0) + 1

    # Loop through each pair of adjacent tokens
    for i in range(len(tokens)-1):
        pair = (tokens[i], tokens[i+1])
        gpt_pair_dict[pair] = gpt_pair_dict.get(pair, 0) + 1

    # Loop through each triplet of adjacent tokens
    for i in range(len(tokens)-2):
        triplet = (tokens[i], tokens[i+1], tokens[i+2])
        gpt_triplet_dict[triplet] = gpt_triplet_dict.get(triplet, 0) + 1

# Prevent possibility of zeros
for tok in unique_tokens:
    human_unique_dict[tok] = human_unique_dict.get(tok, 0) + 1
    gpt_unique_dict[tok] = gpt_unique_dict.get(tok, 0) + 1

for dbl in token_pairs:
    human_pair_dict[dbl] = human_pair_dict.get(dbl, 0) + 1
    gpt_pair_dict[dbl] = gpt_pair_dict.get(dbl, 0) + 1
    
for trip in token_triplets:
    human_triplet_dict[trip] = human_triplet_dict.get(trip, 0) + 1
    gpt_triplet_dict[trip] = gpt_triplet_dict.get(trip, 0) + 1

test_tokens, test_pairs, test_trips = get_tokens([human_test_data, gpt_test_data])
train_tokens, train_pairs, train_trips = get_tokens([human_train_data, gpt_train_data])

oov_pairs = len(test_pairs.difference(train_pairs))/len(test_pairs)
oov_trips = len(test_trips.difference(train_trips))/len(test_trips)
print(f"Bigram OOV rate: {oov_pairs}")
print(f"Trigram OOV rate: {oov_trips}")

# Part ii.c Evaluate model on test set

# Bi/Trigram Classifiers

def bi_class(example):
    tokens = example.split()
    lp_w_g_h  = 0
    lp_w_g_g  = 0
    p_human = len(human_train_data)/(len(human_train_data)+len(gpt_train_data))
    p_gpt = 1 - p_human
    lp_human = math.log(p_human)
    lp_gpt = math.log(p_gpt)

    vocab_size = len(unique_tokens)

    for i in range(1,len(tokens)):
        token = tokens[i]
        pair = (tokens[i-1], tokens[i])
        lp_w_g_h += math.log(human_pair_dict[pair]/(human_unique_dict[token] + vocab_size))
        lp_w_g_g += math.log(gpt_pair_dict[pair]/(gpt_unique_dict[token] + vocab_size))

    if (lp_human + lp_w_g_h) > (lp_gpt + lp_w_g_g):
        return 1
    return 0

def tri_class(example):
    tokens = example.split()
    lp_w_g_h  = 0
    lp_w_g_g  = 0
    p_human = len(human_train_data)/(len(human_train_data)+len(gpt_train_data))
    p_gpt = 1 - p_human
    lp_human = math.log(p_human)
    lp_gpt = math.log(p_gpt)

    vocab_size = len(token_pairs)

    for i in range(2,len(tokens)):
        
        pair = (tokens[i-2], tokens[i-1])
        trip = (tokens[i-2], tokens[i-1], tokens[i])

        lp_w_g_h += math.log(human_triplet_dict[trip]/(human_pair_dict[pair] + vocab_size))
        lp_w_g_g += math.log(gpt_triplet_dict[trip]/(gpt_pair_dict[pair] + vocab_size))

    if (lp_human + lp_w_g_h) > (lp_gpt + lp_w_g_g):
        return 1
    return 0

# Evaluate models
correct = 0
total = 0
for h_example in human_test_data:
    if bi_class(h_example) == 1: correct += 1
    total += 1
for g_example in gpt_test_data:
    if bi_class(g_example) == 0: correct += 1
    total += 1

print(f"Bigram Test Accuracy: {correct/total}")

correct = 0
total = 0
for h_example in human_test_data:
    if tri_class(h_example) == 1: correct += 1
    total += 1
for g_example in gpt_test_data:
    if tri_class(g_example) == 0: correct += 1
    total += 1

print(f"Trigram Test Accuracy: {correct/total}")

# Part iii.a Generate sentences

import numpy as np
def next_tri_word(prior, human):
    T=50
    words = list(unique_tokens)
    probs = np.zeros(len(words))
    w1,w2 = prior
    denominator = 0
    for i,w in enumerate(words):
        trip = (w1,w2,w)
        if human:
            v = math.exp(human_triplet_dict.get(trip,0)/T)
        else:
            v = math.exp(gpt_triplet_dict.get(trip,0)/T)
        probs[i] += v
        denominator += v
    probs = probs/denominator
    return np.random.choice(words, p=probs)

def next_bi_word(prior, human):
    T=50
    words = list(unique_tokens)
    probs = np.zeros(len(words))
    denominator = 0
    for i,w in enumerate(words):
        dbl = (prior,w)
        if human:
            v = math.exp(human_pair_dict.get(dbl,0)/T)
        else:
            v = math.exp(gpt_pair_dict.get(dbl,0)/T)
        probs[i] += v
        denominator += v
    probs = probs/denominator
    return np.random.choice(words, p=probs)

print("\nBIGRAM SENTENCES:\n")
for j in range(5):
    human_seq = ["<START>"]
    gpt_seq = ["<START>"]
    for i in range(20):
        human_seq.append(next_bi_word(human_seq[-1],human=True))
        gpt_seq.append(next_bi_word(gpt_seq[-1], human=False))
    human_seq = " ".join(human_seq[1:])
    gpt_seq = " ".join(gpt_seq[1:])
    print(f"Human {j}: {human_seq}")
    print(f"GPT {j}: {gpt_seq}")
print("\nTRIGRAM SENTENCES:\n")
for j in range(5):
    human_seq = ["<START>", "the"]
    gpt_seq = ["<START>", "the"]
    for i in range(20):
        human_seq.append(next_tri_word((human_seq[-2], human_seq[-1]),human=True))
        gpt_seq.append(next_tri_word((gpt_seq[-2], gpt_seq[-1]), human=False))
    human_seq = " ".join(human_seq[1:])
    gpt_seq = " ".join(gpt_seq[1:])
    print(f"Human {j}: {human_seq}")
    print(f"GPT {j}: {gpt_seq}")
