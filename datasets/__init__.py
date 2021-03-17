from .ssl import *


def select_datasets(name, **kwargs):
    if name == 'ssl_cifar10':
        return SSL_CIFAR10(**kwargs)
