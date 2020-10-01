# COMS6998 MLPP Project
Course Project for COMS6998



## 1. Virtual environment

Let's create a isolated python environment for this project to make sure our code won't conflict due to the different package version. 

Due to the constraints of Pytorch (torch >= 1.5.0 only support python 3.7), I strongly suggest to create a virtual environment in Python 3.7. If you use Anaconda to manage the development environment, you can run the following code to set up the environment:

```conda create -n DLenv python=3.7```

```pip install -r requirements.txt```

(Note that the requirements.txt should in the current file path.)

## 2. About .gitignore
I manually add .idea to the .gitignore. 

## 3. GitHub Action
I have added a ymal in .github/workflows/Lint.yml. It will automatically check our the quality of our codes (Based on Flake8). Feel free to add other pipelines! 

## 4. Create new branches
To avoid unnecessary version conflicts, let's create our own branches when creating modules. Take myself as an example:

```git checkout -b Peimou-dev```

After pushing the code to the remote, create a merge request and wait for QA from other members. (Just to make sure we correctly understand the tasks).

Thanks!
