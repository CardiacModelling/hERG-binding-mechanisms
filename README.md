# Impact of hERG binding mechanisms on risk prediction

This repository contains source code and data that reproduce the article "The impact of uncertainty in hERG binding mechanism on _in silico_ predictions of drug-induced proarrhythmic risk" by Chon Lok Lei, Dominic Whittaker and Gary Mirams.


### Requirements

The code requires Python (3.5+) and the dependencies listed in `requirements.txt`.

To setup, navigate to the path where you downloaded this repo and run
```console
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```


### Outline

- [data](data): Contains all the data, including fitting results.
- [methods](methods): Contains all the Python helper modules, classes and functions.
- [models](models): Model files, contains all the hERG and AP models.
- [protocols](protocols): Contains all the voltage clamp protocols.
- [src](src): Source code for reproducing the results and figures.


## Acknowledging this work

If you publish any work based on the contents of this repository please cite ([CITATION file](CITATION)):

Chon Lok Lei, Dominic G. Whittaker and Gary R Mirams.
The impact of uncertainty in hERG binding mechanism on _in silico_ predictions of drug-induced proarrhythmic risk.

