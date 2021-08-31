import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import spacy
from Transformer import Transformer
from Transformer_Answer import Transformer as Transformer_Answer
from Transformer_Answer_PositionEncoding import Transformer as Transformer_Answer_PE
from translate import greedy_translate, calculate_bleu
from tqdm import tqdm

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets in spacy
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

# Tokenize the German and English texts
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

# Grab the train, val and test data
train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

# Build the vocabulary
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

# Training hyperparams
num_epochs = 100
lr = 3e-4
batch_size = 32

# Model hyperparams
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embed_size = 512
num_heads = 8
num_layers = 3
dropout = 0.1
max_len = 100
multiplier = 4

# Display hyperparams
translate_every = 1
bleu_every = 5

# Translating from German to English
trg_pad_idx = english.vocab.stoi["<pad>"]
src_pad_idx = german.vocab.stoi["<pad>"]

# Print out a few of the source (german) and target (english) keys
print(list(german.vocab.stoi.items())[:10])
print(list(english.vocab.stoi.items())[:10])

# Chunk the data
# Build the train, val, test iterators
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device

)

# define the model
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
    dropout=dropout,
    device=device
).to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

# Define the optimizer and LR scheduler
optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

# quick shape tests to make sure batching is correct

# example_batch = list(train_iterator)[11]
#
# print(example_batch.trg)
#
# print(' '.join([german.vocab.itos[elem] for elem in example_batch.src[5, :]]))
# print(' '.join([english.vocab.itos[elem] for elem in example_batch.trg[5, :]]))

# example sentence: ein marathonläufer läuft an passanten und mobilen toiletten vorbei
# translation: a marathon runner jogs past pedestrians and portable toilets

# start the training loop
# in this codebase, we use the convention [batch_size, seq_len] for src and trg sentences
# the output to the transformer should have shape [batch_size, trg_seq_len, trg_vocab_size]

# example_batch = list(train_iterator)[11]
#
# print(example_batch.trg[:, :-1])
#
# out_batch = model(example_batch.src, example_batch.trg[:, :-1])
# print(out_batch.shape)

# Define an example sentence (here taken from the traning set)
example_src_sentence = "ein marathonläufer läuft an passanten und mobilen toiletten vorbei"

for epoch in range(num_epochs):

    # Greedily translate every translate_every epochs to see how sensible the translation looks
    if epoch % translate_every == 0:
        model.eval()
        translation = greedy_translate(model, example_src_sentence, german, english, device, max_len=50)
        print("==== TRANSLATION ====")
        print(translation)
    
    if epoch % bleu_every == 0:
        model.eval()
        bleu_score = calculate_bleu(model, test_data[:100], german, english, device)
        print(f"bleu on first 100 sentences of test set: {bleu_score:.2f}")

    model.train()
    losses = []

    for _, batch in tqdm(enumerate(train_iterator), total=len(train_iterator)):

        src_batch = batch.src.to(device)
        trg_batch = batch.trg.to(device)

        # Remove end token to feed into model: shape [batch_size, seq_len - 1, trg_vocab_size]
        prediction = model(src_batch, trg_batch[:, :-1])
        prediction = prediction.reshape(-1, prediction.shape[2])
        
        # Reshape to be [batch_size*(seq_len - 1)]

        # Remove start token, want model to be able to predict *next* token
        target = trg_batch[:, 1:].reshape(-1)

        # Zero the gradients
        optimizer.zero_grad()

        # Compute the loss
        loss = criterion(prediction, target)
        losses.append(loss.item())

        # backprop
        loss.backward()

        # do some gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # gradient step
        optimizer.step()

    mean_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch + 1} mean loss: {mean_loss}")




























