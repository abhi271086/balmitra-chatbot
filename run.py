
import os
import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Run Streamlit app on port 8080
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
    stcli.main()
