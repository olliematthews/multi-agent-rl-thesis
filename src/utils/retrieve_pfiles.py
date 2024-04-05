"""
This code will retrieve pickle files from a remote directory. The remote 
directory is at 'remote_path/' + <your current directory name>.
"""

import os

import paramiko

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

# Import any p files
sftp = client.open_sftp()

files = sftp.listdir(remote_path)
for file in files:
    if file.endswith(".p"):
        sftp.get(remote_path + "/" + file, "./" + file)
        print("Imported " + file)
sftp.close()


client.close()
