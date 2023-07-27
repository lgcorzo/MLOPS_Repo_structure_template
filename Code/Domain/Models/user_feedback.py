class Feedback:

    def __init__(self, id, comments, validation):
        self.id = id
        self.comments = comments
        self.validation = validation

    def as_dict(self):
        return {
            'id': self.id,
            'comments': self.comments,
            'validation': self.validation
        }
