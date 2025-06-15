# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier





class KNNConfidence:
    """Baseline which uses disagreement to kNN classifier.
  """

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.kdtree = KDTree(X)
        self.y = y

    def get_score(self, X, y_pred):
        knn_idxs = self.kdtree.query(X, k=self.k)[1]
        knn_outputs = self.y[knn_idxs]
        return np.mean(
            knn_outputs == np.transpose(np.tile(y_pred, (self.k, 1))), axis=1
        )