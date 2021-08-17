# picasso
Code to generate Picasso embeddings of any input matrix as described in .... Picasso maps the points of an input matrix to user-defined, n-dimensional shape coordinates while minimizing reconstruction error using an autoencoder neural network structure. We demonstrate its application to single-cell gene expression matrices.

<p align="center">
  <img src="https://github.com/pachterlab/picasso/blob/main/elExample.png" width="80%" height="80%">
</p>


Getting Started
------------

Examples for running Picasso can be found in the examplePicasso.ipynb. This can be run from Google Colab. Just click on the [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) symbol.

An intro to using Colab can be found [here](https://colab.research.google.com). Briefly, run each code cell by selecting the cell and executing Command/Ctrl+Enter. Code cells can be edited by simply clicking on the cell to start typing.


To run Picasso on your own machine
------------

### Requirements


You need Python 3.6 or later to run Picasso.  You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.

In Ubuntu, Mint and Debian you can install Python 3 like this:

    $ sudo apt-get install python3 python3-pip

For other Linux flavors, macOS and Windows, packages are available at

  https://www.python.org/getit/


### Quick start


Clone this repo:

    $ git clone https://github.com/pachterlab/picasso.git
    $ cd picasso

The necessary environment can be installed:

    $ conda env create -f env/env3.7_LINUX.yml
    $ conda activate env3.7
    
Or for MACOS:

    $ conda env create -f env/env3.7_MACOS.yml
    

Import the module to use as in the examplePicasso.ipynb:

```python
>>> from Picasso import Picasso
```

