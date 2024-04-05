"""
This script will send all the files in the current directory to a directory
at 'remote_path/' + <your current directory name>.
"""

import paramiko
import sys
import os

# Set your username, the host name, your password, and the path to the
# remote directory which contains the directory with the pickle files.
hostname = ""
username = ""
password = ""
remote_path = ""

# Connect to remote host
client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname, username=username, password=password)
remote_path = remote_path + os.path.basename(os.getcwd())

# Setup sftp connection and transmit this script
sftp = client.open_sftp()

try:
    sftp.chdir(remote_path)  # Test if remote_path exists
except IOError:
    sftp.mkdir(remote_path)  # Create remote_path
    sftp.chdir(remote_path)

for root, dirs, files in os.walk("."):
    for file in files:
        # Make sure you only export python files
        if file.endswith(".py"):
            sftp.put(file, file)
            print("Sent " + file)
sftp.close()


client.close()
