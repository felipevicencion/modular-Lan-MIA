import csv
import math
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import re, string
from wordfreq import word_frequency



_PUNC_TABLE = str.maketrans("", "", string.punctuation)

def _normalize_token(w: str) -> str:
    w = w.lower().translate(_PUNC_TABLE)
    w = re.sub(r"\s+", "", w)
    return w


def create_entropy_map(data_sources, dataset_identifier, split, mode='books'):
    """
    Build an entropy map over normalized word forms.
    For English sources, use wordfreq probability proxy p and set entropy = -p*log2(p).
    Unknown words (freq==0) get 0 entropy (treated as very rare).
    """
    vocab = set()

    def add_text(text: str):
        for w in text.split():
            ww = _normalize_token(w)
            if ww:
                vocab.add(ww)

    if mode == 'PILE':
        # data_sources is a (possibly streaming) iterable of dicts with 'text'
        for item in data_sources:
            add_text(item['text'])
    elif mode == 'Dataset':
        print(f"Started processing for mode: {mode}")
        source = dataset_identifier if dataset_identifier is not None else data_sources
        db_data = load_dataset(source, split="train") if isinstance(source, str) else source
        for item in db_data:
            text = item.get('text') if 'text' in item else item.get('snippet', '')
            add_text(text)
    else:
        # fallback: assume iterable of dicts with 'text'
        for item in (data_sources or []):
            add_text(item.get('text', ''))

    # wordfreq returns frequency proxy; use as p for ranking
    entropy_map = {}
    for w in vocab:
        p = word_frequency(w, 'en')  # proxy probability
        entropy_map[w] = (-p * math.log2(p)) if p > 0.0 else 0.0

    return entropy_map


def save_entropy_map(entropy_map, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, entropy in entropy_map.items():
            file.write(f"{word} {entropy}\n")


def load_entropy_map(filename):
    entropy_map = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word, entropy = line.split()
            entropy_map[word] = float(entropy)
    return entropy_map


def sort_entropy_map(entropy_map, descending=True):
    return sorted(entropy_map.items(), key=lambda item: item[1], reverse=descending)


def get_text_files(folder_path):
    return [str(filepath) for filepath in Path(folder_path).rglob('*.txt')]

def load_data_pile(mode, train_val_pile, num_samples=100000):
    if mode == 'PILE':
        data_sources = load_dataset("monology/pile-uncopyrighted", split=train_val_pile, streaming=True)
        if num_samples:
            data_sources = data_sources.take(num_samples)
        return data_sources
    
#changed from tag&tabs implementation to use different datasets
def create_entropy_map_func(dataset_identifier, mode="Dataset", train_val_pile="validation", split):
    if mode == 'Dataset':
        data_sources = [load_dataset(dataset_identifier, split)]
    elif mode == 'PILE':
        data_sources = load_data_pile('PILE', train_val_pile, num_samples=100000)
    print("Creating entropy map...")
    entropy_map = create_entropy_map(data_sources, dataset_identifier, mode=mode)

    filename = "entropy_map.txt"
    print(f"Saving entropy map to {filename}...")
    save_entropy_map(entropy_map, filename)

    print("Loading entropy map...")
    loaded_entropy_map = load_entropy_map(filename)

    print("Sorted Entropy Map:")
    sorted_entropy_map = sort_entropy_map(loaded_entropy_map)
    for word, entropy in sorted_entropy_map[:20]:
        print(f"{word}: {entropy}")

    print("Entropy map loaded and sorted successfully.")
    print("Len of map = ", len(sorted_entropy_map))

    return loaded_entropy_map


def strip_punctuation(word):
    return word.strip(string.punctuation)


def bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY):
    tokens = [_normalize_token(w) for w in line.split()]
    tokens = [t for t in tokens if t]  # drop empties/punct
    top_k = int(TOP_K_ENTROPY)
    # Unknown words => +inf so they won't be incorrectly favored here;
    # truly rare but known (low p) have low entropy and will rank first.
    return sorted(tokens, key=lambda w: entropy_map.get(w, float('inf')))[:top_k]


def create_line_to_top_words_map(text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY,
                                 nlp_spacy):
    doc = nlp_spacy(text)
    all_sentences = list(doc.sents)
    sentences = [s.text.strip() for s in all_sentences
                 if MIN_LEN_LINE_GENERATE <= len(s.text.split()) <= MAX_LEN_LINE_GENERATE]

    line_to_top_words_map = {}

    for line_num, line in enumerate(sentences, 1):
        if not line.strip():
            continue

        # bottom-K low-entropy words from THIS sentence (normalized, no punctuation)
        low_entropy_k = set(bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY))

        # NER spans from THIS sentence, single-token forms normalized (keep multi-token by splitting)
        sent_doc = nlp_spacy(line)
        ner_tokens = set()
        for ent in sent_doc.ents:
            # split multi-word entities into tokens and normalize
            for piece in ent.text.split():
                tok = _normalize_token(piece)
                if tok:
                    ner_tokens.add(tok)

        # Union (final set may exceed K as stated in the paper)
        unique_words = list(low_entropy_k.union(ner_tokens))

        line_to_top_words_map[line_num] = unique_words

    return line_to_top_words_map, sentences

def preprocess_text(text, max_length, tokenizer):
    # Tokenize the text and truncate it to the maximum length
    input_ids = tokenizer.encode(text)
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    return tokenizer.decode(input_ids)

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data


def write_to_csv(data, filename):
    # Check if data is not empty
    if data:
        # Determine the headers from the keys of the first dictionary
        headers = data[0].keys()

        # Create or overwrite the CSV file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in data:

                writer.writerow(row)
