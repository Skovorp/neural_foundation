apt update -y
apt install unzip rsync -y
git config --global user.email "peter.skovorodnikov@gmail.com"
pip install -r <(grep -vE 'torch|numpy' requirements.txt)

# locally run
# rsync -avh --progress /Users/ksc/Downloads/brain_data.zip od_nyu:/root/brain_data.zip

# then run
# unzip brain_data.zip
