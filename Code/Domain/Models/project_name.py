import uuid


class ProjectName:

    def __init__(self, result):
        self.uid = uuid.uuid4()
        self.metric = result['metric'].round(2)
        self.post = result['post']  # Si las pandas series estan vacias ->result['post'][0]
        self.machine = result['machine']  # Si las pandas series estan vacias ->result['machine'][0]
        self.file = result['file']

    def as_dict(self):
        return {
            'id': self.uid,
            'metric': self.metric.to_json(),
            'machine': self.machine.to_json(),
            'post EXE File': self.post.to_json()
        }
