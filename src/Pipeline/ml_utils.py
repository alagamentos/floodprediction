from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def _subplot_cm(cm,
        classes,
        fig, ax,
        normalize=False,
        title=None):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.

  :type cm: array
  :param cm: confusion matrix array

  :type classes: list
  :param classes: list containing label strings

  :type normalize: boolean
  :param normalize: normilize by rows

  :type title: string
  :param title: plot title, defaults to None
  """

  if not title:
    if normalize:
      title = 'Normalized'
    else:
      title = 'Without normalization'

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

  ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    title=title,
    ylabel='True label',
    xlabel='Predicted label')

  plt.setp(ax.get_xticklabels(),
       rotation=45,
       ha="right",
       rotation_mode="anchor")

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j,
          i,
          format(cm[i, j], fmt),
          ha="center",
          va="center",
          color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()

def plot_confusion_matrix(y_pred, y_true, labels, suptitle = 'Confusion Matrix'):
  """
  _subplot_cm wapper - Plot normalized and not
  normilized confusion matrix

  :type cm: array
  :param cm: confusion matrix array

  :type labels: list
  :param labels: list containing label strings

  :type suptitle: string
  :param suptitle: plot title, defaults to Confusion Matrix

  """

  cm = confusion_matrix(y_pred, y_true)

  fig, ax = plt.subplots(1,2, sharey=True)
  fig.suptitle(suptitle)
  _subplot_cm(cm, labels ,fig, ax[0], normalize=False)
  _subplot_cm(cm, labels ,fig, ax[1], normalize=True)
  plt.show()
