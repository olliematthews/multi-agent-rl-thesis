# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:52:27 2019

@author: Ollie
"""

import paramiko
import os

# Connect to remote host
client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('', username='', password='')
remote_path = '' + os.path.basename(os.getcwd())

# Import any p files
sftp = client.open_sftp()

files = sftp.listdir(remote_path)
for file in files:
    if file.endswith('.p'):
        sftp.get(remote_path + '/' + file, './' + file)
        print('Imported ' + file)
sftp.close()



client.close()
