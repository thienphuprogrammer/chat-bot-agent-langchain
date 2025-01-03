import logging


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseSingleton(metaclass=Singleton):
    pass


class BaseObject(BaseSingleton):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.class_name())

    @classmethod
    def class_name(cls):
        return cls.__name__
