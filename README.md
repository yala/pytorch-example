# pytorch-example
Example of setup of a pytorch project.
This was part of a lecture for MIT's Adv Nat Language Processing course,
6.864.

## Project Structure
- All neural net code is setup as module for convienece
    - data: defines dataset object and its utils
    - models: define models and their utils
    - train: defines code for training network and logging perfromance
- Run all scripts from a script directory
    - main.py: entry point to run model
- requirements.txt
    - requirements to run this code on a linux machine with CUDA 8. For
    alternate OS's, please check pytorch documentation.


