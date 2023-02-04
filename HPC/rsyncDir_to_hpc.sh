#!/bin/sh


echo "Setting up configuration..."
sshLogin="amosc@login1.hpc.dtu.dk"

LocalDir="/Users/amosc/Documents/energycommunityplanning/"
RemoteDir="~/Documents/energycommunityplanning/"


echo "Transferring files to HPC..."
rsync -avh --exclude '*.git' --exclude 'data/out' --exclude '__pycache__' ${LocalDir}/ $sshLogin:$RemoteDir
