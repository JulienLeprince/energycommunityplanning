#!/bin/sh


echo "Setting up configuration..."
sshLogin="amosc@login1.hpc.dtu.dk"

LocalDir="/Users/amosc/Documents/energycommunityplanning/data/out/"
RemoteDir="~/Documents/energycommunityplanning/data/out/"


echo "Transferring file from HPC..."
rsync -avh --exclude '*.err' --exclude '*.out' --exclude '*.git' -e ssh $sshLogin:$RemoteDir $LocalDir
