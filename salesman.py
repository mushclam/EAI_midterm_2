import sys

class Salesman:
    def __init__(self, filename):
        self.filename = filename
        self.location = []

    def read(self):
        try:
            with open(self.filename, 'r') as f:
                lines = f.read().splitlines()
                for item in lines:
                    if len(item) == 0:
                        continue
                    pair = item.split(' ')
                    point = (int(pair[0]), int(pair[1]))
                    self.location.append(point)
        except FileNotFoundError as e:
            print(e)

    def print(self):
        print(self.location)