#from: https://stackoverflow.com/questions/71500106/how-to-implement-t-sne-in-tensorflow

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2).fit_transform(features)