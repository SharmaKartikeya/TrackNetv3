class BaseAgent:
    def __init__(self, config):
        self.config = config

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def training_loop(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError
    
    def validation_loop(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError