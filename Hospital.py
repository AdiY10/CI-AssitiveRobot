from Patient import Patient
from Room import Room
import pandas as pd
import math
import matplotlib.pyplot as plt

######## READ DATA ###########
Patient_data = pd.read_csv('Patient.csv')
Rooms_data = pd.read_csv('Rooms.csv')

######## Variables ##########
robot_speed = 6


def init_rooms(Rooms_data):
    room_dict = {}
    for room in Rooms_data.iterrows():
        room_dict[room[1]['id']] = Room(room[1]['id'],room[1]['x'],room[1]['y'])
    return room_dict


def init_patient(Patient_data,r_data):
    patient_array = []
    for patient in Patient_data.iterrows():
        patient_array.append(Patient(patient[1]['id'],r_data[patient[1]['room']],patient[1]['urgency'], patient[1]['typeofdisease']))
    return patient_array

def calc_dist(room1 , room2):
    return(math.sqrt(((room1[0] - room2[0])**2) + ((room1[1] - room2[1])**2)))



def distance_matrix(room_data):
    print(room_data)
    df = pd.DataFrame(columns=range(1,len(room_data)+1), index=range(1,len(room_data)+1))
    for i in room_data.keys():
        for j in room_data.keys():
            df.at[i,j] = calc_dist(room_data[i].get_loc(),room_data[j].get_loc())
    return df

def plotonimage(room_data):
    x_arr = []
    y_arr = []
    for room in room_data:
        1264




if __name__ == '__main__':
    room_data = init_rooms(Rooms_data)
    patient_data = init_patient(Patient_data, room_data)
    dist = distance_matrix(room_data)

