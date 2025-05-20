from tqdm import tqdm
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn, HW=224*224, n_classes=1000):
        r"""Create deletion/insertion metric instance.

        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.HW = HW
        self.n_classes = n_classes

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        img_batch = img_batch.cpu()
        n_samples = img_batch.shape[0]
        predictions = torch.FloatTensor(n_samples, self.n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = self.model(
                img_batch[i*batch_size:(i+1)*batch_size].to(device)).detach().cpu()
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions, -1)
        n_steps = (self.HW + self.step - 1) // self.step
        scores = np.empty((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(
            exp_batch.reshape(-1, self.HW), axis=1), axis=-1)

        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(img_batch)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(
                img_batch[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = img_batch.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'pixels'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = self.model(
                    start[j*batch_size:(j+1)*batch_size].to(device))
                preds = preds.detach().cpu().numpy()[range(
                    batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            if i < n_steps:
                for rr in r:
                    start.cpu().numpy().reshape(n_samples, 3, self.HW)[rr, :, coords.reshape(n_samples, 3, n_steps)[
                        rr]] = finish.cpu().numpy().reshape(n_samples, 3, self.HW)[rr, :, coords.reshape(n_samples, 3, n_steps)[rr]]
        print('AUC: {}'.format(auc(scores.mean(1))))
        return scores.transpose()
