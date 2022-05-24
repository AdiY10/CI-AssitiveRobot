
class Patient(object):

    def __init__(self, id, room, urgency, type_of_disease):
        self.id = id
        self.room = room
        self.urgency = urgency
        self.type_of_disease = type_of_disease

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def __hash__(self):
        return self.id

