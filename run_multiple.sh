#!/bin/bash
BASEDIR=$(dirname $0)
echo "Script location: ${BASEDIR}"
cd ${BASEDIR}

python3 run.py -c ./config_cp_200.yml
rm -r ./checkpoints
python3 run.py -c ./config_cp_200.yml
rm -r ./checkpoints
