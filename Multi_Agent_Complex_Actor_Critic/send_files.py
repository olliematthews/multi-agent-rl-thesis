import paramiko
import sys
import os

# Connect to remote host
client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('', username='', password='')

# Setup sftp connection and transmit this script
sftp = client.open_sftp()
remote_path = '' + os.path.basename(os.getcwd())

try:
    sftp.chdir(remote_path)  # Test if remote_path exists
except IOError:
    sftp.mkdir(remote_path)  # Create remote_path
    sftp.chdir(remote_path)

for root, dirs, files in os.walk("."):
    for file in files:
        # Make sure you only export python files
        if file.endswith('.py'):
            sftp.put(file, file)
            print('Sent ' + file)
sftp.close()


client.close()

