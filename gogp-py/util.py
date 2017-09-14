import time

class Util:
    @staticmethod
    def get_string_time(padding=''):
        return time.strftime("_%y%m%d_%H%M%S",time.localtime()) + padding