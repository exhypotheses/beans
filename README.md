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

**Environment**

This repository's virtual environment is created via

```shell
    conda env create -f environment.yml -p /opt/miniconda3/envs/uncertainty
```

To generate the dotfile that [`pylint`](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html) - the static code analyser - will use for code analysis, run

```shell
    pylint --generate-rcfile > .pylintrc
```

and edit `.pylintrc` if necessary.  Subsequently, within `vscode` settings ensure that the `Pylint: Import Strategy` is

> fromEnvironment

<br>
<br>

### Modelling Notes

**Independent Variables & Dimensionality Reduction**

In the case of dimension reduction via principal component analysis, the key consideration is the effective number of  principal components.  A rough estimate can be determined via the elbow method, however beware of the method's limitations.  Study

* [Estimating the number of clusters in a data set via the gap statistic](https://statweb.stanford.edu/~gwalther/gap)
* [The Application of Cluster Analysis in Strategic Management Research: An Analysis and Critique](https://www.jstor.org/stable/2486927?seq=1)


<br>

**Automatic Differentiation Variational Inference & Mini Batches**

Whenever an algorithm has a mini-batch option, try it.  Study [Variational Inference: Bayesian Neural Network](https://www.pymc.io/projects/examples/en/latest/variational_inference/bayesian_neural_network_advi.html#mini-batch-advi) and

* [pymc.Minibatch()](https://www.pymc.io/projects/docs/en/latest/api/generated/pymc.Minibatch.html)

* [example](https://www.pymc.io/projects/examples/en/latest/variational_inference/variational_api_quickstart.html#minibatches)

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