# tensorpic
Experiments with TensorFlow for image classification and convolutional neural network and optimization of such.
Includes the export and running of a deployable TensorFlow Lite model.

Keras will download the datasets into ~/.keras/datasets/

1. See the setup script for details on setting up Python and the Virtual Environment using Pyenv.
   setup-project.zsh
2. Use the setup script to update pip and setuptools and to install from requirements.txt
3. Run the script: tf-convo-neural-net-cifar.py (PROJECT 1)
4. Run the script: tf-image-classification-flowers.py (PROJECT 1)

Execution will pause when plots are show. Close the plot window to resume program execution.

(This file is only a stub for an upcoming project to be completed later: tf-performance-tune-bench.py)


PYENV:
I very strongly recommend using Pyenv. Python developers work on a lot of projects at once. It is important to match
the Python version and all module versions (which can depend on the Python version.) Ideally all these versions which
you will run, deploy or continue development with should exactly match those under which your application code was
developed and tested. This is a fundamental concept which most developers try to or must follow. If this is not
followed, bugs and problems can arise. There really is no practical alternative to using Pyenv to easily do this.
With Pyenv you really do get all the flexibility and control you must have in this area, and it is lightweight and
completely cross-platform. I also consider Pyenv Virtualenv to be the easiest and best way to manage Python
virtual environments. I would even say Pyenv is fun and rewarding to use. You can install and use almost ANY version
or flavor of Python you like, compiled from source for your platform, on your platform. Then you can switch to it
quickly and naturally. Your virtual environments will activate automatically using .python-version, which is a
fantastic feature. I am sure you will find the benefits to your workflow significant. And you will never have to
worry that you were not using the correct version of Python or the correct version of any module when there
are issues to address.

The setup script assumes you are using Pyenv and that you have installed the closest-matching Python version
and have created the Pyenv Virtualenv of the correct name so that this projects .python-version file
takes immediate effect.

