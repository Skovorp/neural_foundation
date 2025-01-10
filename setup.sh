apt update -y
apt install zip unzip pigz pv -y
git config --global user.email "peter.skovorodnikov@gmail.com"
pip install -r <(grep -vE 'torch|numpy' requirements.txt)
# mkdir /root/giant_cache
cp /workspace/full_clean.zip /root/full_clean.zip
cd /root
unzip full_clean.zip
wandb init

# locally run
# rsync -avh --progress /Users/ksc/Downloads/brain_data.zip od_nyu:/root/brain_data.zip

# then run
# unzip brain_data.zip

# parallelly run
# pv --size $(stat --format=%s giant_data.tar.gz) giant_data.tar.gz | pigz -p 2 -dc | tar -xf - -C /root
