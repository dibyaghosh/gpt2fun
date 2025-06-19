echo ssh $1 "sudo chown dibya ~/nfs/pretraining/"
ssh $1 "sudo chown dibya ~/nfs/pretraining/ && sudo chown -R dibya ~/nfs/pretraining/src/"
echo rsync -r --info=progress2 /home/dibya/nfs/pretraining/ $1:~/nfs/pretraining/
rsync -r --info=progress2 --exclude .venv --exclude env --exclude wandb --exclude notebooks --exclude .micromamba /home/dibya/nfs/pretraining/ $1:~/nfs/pretraining/