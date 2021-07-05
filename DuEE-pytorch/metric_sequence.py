import numpy as np
import torch

class Accuracy():

    def __init__(self):

        #self.id2label_dict = dict(enumerate(label_list))
        self.num_infer = 0
        self.num_label = 0
        self.num_correct = 0

    def compute(self, predictions, labels):
        """
        Compute the top-k (maxinum value in `topk`) indices.

        Args:
            pred (Tensor): The predicted value is a Tensor with dtype
                float32 or float64. Shape is [batch_size, d0, ..., dN].
            label (Tensor): The ground truth value is Tensor with dtype
                int64. Shape is [batch_size, d0, ..., 1], or
                [batch_size, d0, ..., num_classes] in one hot representation.

        Return:
            Tensor: Correct mask, a tensor with shape [batch_size, topk].
        """
        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
        predictions = np.argmax(predictions, axis=1)
        self.num_correct += sum(predictions==labels)
        self.num_label +=len(labels)

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.num_infer = 0
        self.num_label = 0
        self.num_correct = 0

    def accumulate(self):
        """
        Computes and returns the accumulated metric.
        """
        precision = float(
            self.num_correct /
            self.num_label)

        return precision