import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
from torch.utils.tensorboard import SummaryWriter
from gtts import gTTS
import os
from moviepy.editor import concatenate_audioclips, AudioFileClip
import re
from pydiogment.augt import speed


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get characters from string.printable
all_characters = re.sub(r"[A-Z]+", r"", string.printable)
n_characters = len(all_characters)

# Read large text file (Note can be any text file: not limited to just names)
file = unidecode.unidecode(open("/media/larry/mydiskishuge/code/autotainment/output.txt").read())


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class Generator:
    def __init__(self):
        self.chunk_len = 1000
        self.num_epochs = 50000
        self.batch_size = 1
        self.print_every = 1
        self.hidden_size = 512
        self.num_layers = 3
        self.lr = 0.003
        self.rnn = RNN(
            n_characters, self.hidden_size, self.num_layers, n_characters
        ).to(device)
        self.rnn.load_state_dict(torch.load("/media/larry/mydiskishuge/code/autotainment/model.pth"))

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    def generate(self, initial_str, predict_len, temperature=0.9):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(
                initial_input[p].view(1).to(device), hidden, cell
            )

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(
                last_char.view(1).to(device), hidden, cell
            )
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted

    # input_size, hidden_size, num_layers, output_size
    def train(self):
        self.rnn.load_state_dict(torch.load("/media/larry/mydiskishuge/code/autotainment/model.pth"))

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f"runs/names0")  # for tensorboard

        print("=> Starting training")

        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f"Loss: {loss}")
                print(self.generate(initial_str="yo", predict_len=200, temperature=1))
                torch.save(self.rnn.state_dict(), "/media/larry/mydiskishuge/code/autotainment/model.pth")

            writer.add_scalar("Training loss", loss, global_step=epoch)


pack = Generator()

with open("/media/larry/mydiskishuge/code/autotainment/lines.txt") as lines:
    l = lines.readlines()
    for i in range(0, len(l)):
        output = pack.generate(initial_str=l[i], predict_len=250, temperature=0.75)

        output = re.sub(r"[\n.,!?;:'\"]+", r"", output)

        print(output)

        name = f"{random.randint(0, 999999999999999999999999)}.mp3"

        os.system(f"python3 /media/larry/mydiskishuge/code/autotainment/tiktok-voice/main.py -v en_us_001 -t \"{output}\" -n /media/larry/mydiskishuge/code/autotainment/output/audio/{name} --session 44982e2ae95ee9a3031bc6a6d47a325b")

        test_file = f"/media/larry/mydiskishuge/code/autotainment/output/audio/{name}"