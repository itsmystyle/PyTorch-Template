from module.model import VGG16


def create_model(args, num_class):
    return VGG16(n_classes=num_class)
