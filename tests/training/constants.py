from lycil.backbone import ConvNetArgs

# dataset constants
CIFAR10_PATH = "data/cifar10"
CIFAR10_LABEL_COL = "label"

CIFAR100_PATH = "data/cifar100"
CIFAR100_LABEL_COL = "fine_label"

# dataloader kwargs
VAL_LOADER_KWARGS = {"batch_size": 128, "shuffle": False, "num_workers": 10}
TEST_LOADER_KWARGS = VAL_LOADER_KWARGS

CONVNET_ARGS = {
    # not dummy
    False: ConvNetArgs(name="resnet50", pretrained=True, cifar=True),
    # dummy training, for fast tests
    True: ConvNetArgs(name="resnet18", pretrained=False, cifar=True),
}
