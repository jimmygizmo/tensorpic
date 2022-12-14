#! /usr/bin/env zsh

# Use all or just some of the below commands as needed.
# It is not guaranteed you can just run this script as is.
# You need Pyenv and Pyenv Virtualenv and you need to be inside the project root.
# Then you might be able to use this entire script as is.
# What might be most convenient is to just run this from the pip installs. So I have disabled the Pyenv steps.
# I'm sure you will figure it out. :)

# 1. install latest Python 3. I'm using 3.10.6.  #### DISABLED. Uncomment if you need it.
####pyenv install 3.10.6

# 2. create the virtual environment using the name : ve.textblob
####pyenv virtualenv 3.10.6 ve.tensorpic  #### DISABLED. Uncomment if you need it.

# And now the ve.tensorpic virtual env should be active if all was done correctly.
# pip/python are now those within the VE.

# Always upgrade pip and setuptools in a fresh virtual environment. They always need it.
pip install --upgrade pip
pip install --upgrade setuptools

# Now we can install the modules we will need.
# The exact version of every module will match those used when the project code was last updated and tested.
# If you want or need to let Python/pip install the latest available versions of everything,
# then use requirements.txt as this file does not pin versions but will still provide everything you need.
pip install -r pinned-requirements.txt


# ** Be sure to set the correct project interpreter in your IDE. I use PyCharm and it has just a little bit of
# trouble automatically finding the right interpreter, but after a little fiddling I can manually set it.
# IntelliJ IDE teams, please make your amazing IDEs I use and love so much full compatible with Pyenv and
# Pyenv Virtualenv. :)
# This is the path I set it to on my Mac. It took a few tries because the IDE even unhelpfully translated
# the final symlink, which was incorrect, but on the second try it accepted the path I entered:
# /Users/your_username/.pyenv/versions/ve.tensorpic/bin/python

