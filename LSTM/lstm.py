""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy as np
import tensorflow as tf
from keras import backend as K
from music21 import converter, instrument, note, chord

from keras.models import Sequential
import keras
from keras_self_attention import SeqSelfAttention
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Bidirectional
from keras.layers import Activation

from tensorflow.keras import utils
from keras.callbacks import ModelCheckpoint


# disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()
    #with open('data/notes', 'rb') as filepath:
    #    notes = pickle.load(filepath)

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the .mid files"""
    notes = []

    for file in glob.glob("classical/*.mid"): #("jazz_processed/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    print("n_vocab: ", n_vocab)
    """ create the structure of the neural network """
    model = Sequential()
    model.add(keras.Input((network_input.shape[1],
                           network_input.shape[2])))
    model.add(Bidirectional(LSTM(
        512,
        recurrent_dropout=0.3,
        return_sequences=True
    )))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    #model.add(Attention(use_scale=True)())
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "data/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='auto',
        save_freq=64
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=50, batch_size=32, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
    