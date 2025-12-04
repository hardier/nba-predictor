#!/bin/bash

# 1. Run the Scheduler in the background (&)
# We use 'nohup' to ensure it keeps running
nohup python scheduler.py > scheduler.log 2>&1 &

# 2. Run the Streamlit App in the foreground
# Streamlit needs to bind to the port Render provides ($PORT)
streamlit run app.py --server.port $PORT --server.address 0.0.0.0