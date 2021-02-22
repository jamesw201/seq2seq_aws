import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

from Multi30K_S3.Multi30k import Multi30k
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

torch.cuda.is_available()

# Copy inference pre/post-processing script so that it'll be included in the model package
os.system('mkdir /opt/ml/model/code')
# os.system('cp inference.py /opt/ml/model/code')
os.system('cp requirements.txt /opt/ml/model/code')

# TODO: how do we get these from the S3 parameters?
spacy_eng = spacy.load('en_core_web_sm')
spacy_ger = spacy.load('de_core_news_sm')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        
        return hidden, cell
    
    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)
        # grab start token
        x = target[0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
            
        return outputs


# Model hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)


def main(args):
    # Tensorboard
    writer = SummaryWriter(f'runs/loss_plot')
    step = 0
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
       (train_data, validation_data, test_data),
        batch_size=args.batch_size,
        sort_within_batch=True,
        sort_key = lambda x: len(x.src),
        device=device
    )

    encoder_net = Encoder(input_size_encoder, args.encoder_embedding_size,
                         args.hidden_size, args.num_layers, args.enc_dropout).to(device)

    decoder_net = Decoder(input_size_decoder, args.decoder_embedding_size,
                         args.hidden_size, output_size, args.num_layers, args.dec_dropout).to(device)

    model = Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    pad_idx = english.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if args.load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

    sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

    for epoch in range(args.epochs):
        print(f'Epoch [{epoch} / {args.epochs}]')
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

        model.eval()

        translated_sentence = translate_sentence(
            model, sentence, german, english, device, max_length=50
        )

        print(f"Translated example sentence: \n {translated_sentence}")

        model.train()

        for batch_idx, batch in enumerate(train_iterator):
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            output = model(inp_data, target)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            writer.add_scalar('Training loss', loss, global_step=step)
            step += 1

    # Save model to model directory
    torch.save(model, f'{args.model_output_dir}/model.pth')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--epochs',        type=int,     default=20)
    parser.add_argument('--learning-rate', type=float,   default=0.001)
    parser.add_argument('--batch-size',    type=int,     default=64)
    parser.add_argument('--load-model',    type=bool,    default=False)
#     parser.add_argument('--weight-decay',  type=float, default=2e-4)
#     parser.add_argument('--momentum',      type=float, default='0.9')
    parser.add_argument('--optimizer',     type=str,     default='adam')
    parser.add_argument('--enc-dropout',   type=float,   default=0.5)
    parser.add_argument('--dec-dropout',   type=float,   default=0.5)
    parser.add_argument('--num-layers',    type=int,     default=2)
    parser.add_argument('--hidden-size',   type=int,     default=1024)
    parser.add_argument('--encoder-embedding-size',   type=int,   default=300)
    parser.add_argument('--decoder-embedding-size',   type=int,   default=300)

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--model_output_dir', type=str,   default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--gpu-count',        type=int,   default=os.environ['SM_NUM_GPUS'])
#     parser.add_argument('--training',         type=str,   default=os.environ['SM_CHANNEL_TRAIN'])
#     parser.add_argument('--validation',       type=str,   default=os.environ['SM_CHANNEL_VALIDATION'])
#     parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])
    
    args = parser.parse_args()
    main(args)
