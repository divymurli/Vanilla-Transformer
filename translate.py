import torch
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import spacy

from Transformer import Transformer

### PREPROCESSING USING TORCHTEXT ###
### NOTE: The below is only meant for testing, the main function in this script is greedy_translate which greedily decodes the transformer's output
### Load the datasets ###

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    """
    Tokenize German text
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenize English text
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


german = Field(
            tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True
)

english = Field(
            tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True
)

### grab the train, val and test data using torchtext ###

train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

# build the vocabulary
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

### Define the greedy decoding function ###
def greedy_translate(model, example_src_sentence, src_field, trg_field, device, max_len=50):

    # for now src field is German
    spacy_de = spacy.load('de')

    # tokenize
    tokenized_src_text = [tok.text.lower() for tok in spacy_de(example_src_sentence)]

    # prepend and append start and end tokens to sentence
    tokenized_src_text.insert(0, src_field.init_token)
    tokenized_src_text.append(src_field.eos_token)

    # convert tokens to indices
    tokenized_src_indices = [german.vocab.stoi[tok] for tok in tokenized_src_text]

    # convert to torch_long_tensor of shape [batch_size, seq_len] (batch_size is 1 here)
    src_tensor = torch.LongTensor(tokenized_src_indices).unsqueeze(0).to(device)
    # print("src_tensor", src_tensor)

    # start translated sentence with the start token
    translated_sentence = [trg_field.vocab.stoi["<sos>"]]

    # initiate a counter which increments till the full sentence is translated
    # i.e. when the model guesses the final token
    counter = 0
    while counter < max_len:

        trg_tensor = torch.LongTensor(translated_sentence).unsqueeze(0).to(device)
        with torch.no_grad():

            output = model(src_tensor, trg_tensor)

            # grab the last prediction index
            prediction = output.argmax(2)[:, -1].item()
            translated_sentence.append(prediction)

        if prediction == trg_field.vocab.stoi["<eos>"]:
            break

        counter += 1
    
    print(f"Translated tokens: {translated_sentence}")

    translated_sentence = [trg_field.vocab.itos[idx] for idx in translated_sentence]

    return translated_sentence

# Meant for testing purposes only

if __name__ == "__main__":

    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    example_src_sentence = "ein marathonläufer läuft an passanten und mobilen toiletten vorbei"

    trg_pad_idx = english.vocab.stoi["<pad>"]
    src_pad_idx = german.vocab.stoi["<pad>"]

    # hparams
    src_vocab_size = len(german.vocab)
    trg_vocab_size = len(english.vocab)
    embed_size = 512
    num_heads = 8
    num_layers = 3
    dropout = 0.1
    max_len = 100
    multiplier = 4

    model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    embedding_dim=embed_size,
    max_len=max_len,
    heads=num_heads,
    num_layers=num_layers,
    multiplier=multiplier,
    dropout=dropout
).to(device)

    print(greedy_translate(model, example_src_sentence, german, english, device))










