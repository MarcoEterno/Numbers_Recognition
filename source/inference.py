from config import get_system_device
from image_classifier import ImageClassifier


def test_model_performance(clf: ImageClassifier, test_datasets, device=get_system_device()):
    clf.eval()
    x, y = next(iter(test_datasets))
    x = x.to(device)
    y = y.to(device)
    output = clf(x)
    accuracy = (output.argmax(1) == y).sum().item() / len(y)
    print("model accuracy on test dataset is: ", accuracy)
    return accuracy