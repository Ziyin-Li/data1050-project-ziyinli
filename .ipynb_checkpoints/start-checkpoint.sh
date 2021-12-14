#!/usr/bin/env bash
set -xe

mkdir -p /var/log;
chmod -R 777 /var/log;
python3 app.py;
