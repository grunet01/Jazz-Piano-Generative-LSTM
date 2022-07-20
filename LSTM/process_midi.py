import os
import glob
import pickle
from mido import MidiFile


def process_midi_data():
    file_index = 0
    
    for file in glob.glob("jazz/*.mid"):
        file_index += 1
        
        print("============================================================")
        print("Parsing: " + file)
        mid = MidiFile(file, clip=True)
        print(mid)
        
        num_messages_list = []
        for track in mid.tracks:
            num_messages_list.append(get_num_messages(track))
        max_index = num_messages_list.index(max(num_messages_list))
        
        new_mid = MidiFile(type=0, clip=True)
        new_mid.tracks.append(mid.tracks[max_index])
        track_name = "jazz_processed/" + str(file_index) + ".mid"
        new_mid.save(track_name)
        print("Track Index " + str(max_index) + " SAVED as " + track_name)
        
        
    print("============================================================")
    print("Parsed " + str(file_index) + " files successfully")
    print("============================================================")

def print_track_data():
    for file in glob.glob("jazz_processed/*.mid"):
        print("============================================================")
        print("Parsing: " + file)
        mid = MidiFile(file, clip=True)
        print(mid)
        print(mid.tracks)
    print("============================================================")        
    print("Printed " + str(len(glob.glob(("jazz_processed/*.mid")))) + " files")
    print("============================================================")        

def get_num_messages(track):
    track_string = str(track)
    number = [int(word) for word in track_string.split() if word.isdigit()]
    return number[0]


if __name__ == '__main__':
    print_track_data()
    #process_midi_data()