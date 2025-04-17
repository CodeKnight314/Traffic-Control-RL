pip install -r requirements.txt
add-apt-repository ppa:sumo/stable
apt-get update
apt-get install sumo sumo-tools sumo-doc
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
apt-get install tmux