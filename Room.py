
class Room(object):

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.id == other.id

    def __hash__(self):
        return self.id

    def __str__(self):
        return str(self.id)

    def get_loc(self):
        return (self.x,self.y)


