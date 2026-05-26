class BaseEvaluator:
    def __init__(self, model_name, data_path):
        self.model_name = model_name
        self.data_path  = data_path

    def run(self):
        raise NotImplementedError("Each evaluator must implement run()")
