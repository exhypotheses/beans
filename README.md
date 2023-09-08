<br>

This repository is focused on the [Dry Beans Data](http://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset) of Murat 
Koklu & Ilker Ali Ozkan .  A selection of the beans will be used to explore audit options.

<br>

References:
* _Murat Koklu, Ilker Ali Ozkan: Multiclass classification of dry beans using computer vision and machine learning techniques, Computers and Electronics in Agriculture, 
  Volume 174, 2020, 105507_, [DOI](https://doi.org/10.1016/j.compag.2020.105507)
* [The details of the original dataset](http://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
* [The original zip file](http://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip)
* [The developer's copy of this original zip file](https://github.com/thirdreading/hub/blob/master/data/beans/beans.zip)

<br>
<br>

### Development Notes

<br>

**Environment**

Refer to the [github.com/briefings/energy Development Notes](https://github.com/briefings/energy#development-notes), it outlines the
creation & usage of the environment `miscellaneous`, which is used by this repository also.

<br>

**Requirements**

```bash
    conda activate miscellaneous
    pip freeze -r docs/filter.txt > requirements.txt
```

whereby filter.txt does not include `python-graphviz`, `pywin32`, `nodejs`.  And, w.r.t. conventions

```bash
    pylint --generate-rcfile > .pylintrc
```

Subsequently, within `vscode` settings ensure that the `Pylint: Import Strategy` is

> fromEnvironment

<br>
<br>

### Modelling Notes

<br>

**Projecting the indepent variables**

Foremost, how many principal components might be effective?  A rough estimate can be determined via the elbow method, whilst being aware of the method's limitations.  Study

* [Estimating the number of clusters in a data set via the gap statistic](https://statweb.stanford.edu/~gwalther/gap)
* [The Application of Cluster Analysis in Strategic Management Research: An Analysis and Critique](https://www.jstor.org/stable/2486927?seq=1)

And

* Always add the 'kpca' projector object to the preprocessing pickle; if a projection step is included in the modelling process.

```
  def project_(self, training_scaled: pd.DataFrame):
    """
  
    :param training_scaled:  The data that will be projected
    :return:
    """
  
    knee = beans.functions.knee.Knee()
    n_components = knee.exc(blob=training_scaled, target=self.target)
  
    project = beans.functions.project.Project()
    matrix = training_scaled.drop(columns=self.target).to_numpy()
    projector = project.exc(matrix=matrix, n_components=n_components)
    training_projected = project.apply(matrix=matrix, vector=training_scaled[self.target], 
                                       projector=projector)
  
    return training_projected, projector
```

```
  logger.warning('The # of Kernel Principal Components of interest will be {}'.format(n_components))
  logger.warning(training_projected.info())
```

<br>
<br>


### References

* [imbalanced learn (imblearn)](https://imbalanced-learn.org/stable/index.html)
  * [SVMSMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SVMSMOTE.html)

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>