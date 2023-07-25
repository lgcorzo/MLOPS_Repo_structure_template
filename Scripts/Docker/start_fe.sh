#!/bin/bash
gunicorn Code.FrontEnd.app:server --workers 4 --threads 2 --bind 0.0.0.0:8001 &