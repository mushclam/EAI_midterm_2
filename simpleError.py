class DatasetError(Exception):
    def __str__(self):
        return "Dataset is not proper."

class CapacityNotExistError(Exception):
    def __str__(self):
        return "Total weight capacity is not exist."