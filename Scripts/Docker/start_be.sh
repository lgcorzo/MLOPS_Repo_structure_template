#!/bin/bash
gunicorn Code.Controller.app:app --workers 4 --threads 2 --bind 0.0.0.0:8000 &