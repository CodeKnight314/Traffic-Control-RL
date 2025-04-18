pip install -r requirements.txt
add-apt-repository ppa:sumo/stable
apt-get update
apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME="/usr/share/sumo"
apt-get install tmux xvfb