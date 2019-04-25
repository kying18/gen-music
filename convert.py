from music21 import converter, instrument, note, chord
import glob
import numpy as np
from keras.utils import np_utils
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

def get_notes():
    notes = []
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    print('Attained all notes in string format')

    return notes

def get_network_inp_out(notes, seq_length=100):
    # sort pitches
    pitches = sorted(set(note for note in notes))
    n_vocab = len(pitches)

    # map each pitch to a number
    # TODO: look into mappiing pitches to numbers...
    note_to_num = dict((note, number) for number, note in enumerate(pitches))

    network_input = []
    network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - seq_length, 1):
        sequence_in = notes[i:i + seq_length]
        sequence_out = notes[i + seq_length]
        network_input.append([note_to_num[char] for char in sequence_in])
        network_output.append(note_to_num[sequence_out])
    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, seq_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output

def train_model(net_in, n_vocab):
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(net_in.shape[1], net_in.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


