apt update -y
apt install unzip -y
git config --global user.email "peter.skovorodnikov@gmail.com"
pip install -r <(grep -vE 'torch|numpy' requirements.txt)

scp od_nyu:/Users/ksc/Downloads/neural_data.zip .
unzip neural_data.zip -d ./neural_data
