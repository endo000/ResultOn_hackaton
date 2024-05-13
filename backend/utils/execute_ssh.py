import paramiko
import os
from dotenv import load_dotenv




# method to make enviroment variables avalable
load_dotenv()

# method to execute a command via ssh
def execute_ssh_command(command):
    try:

        # getthe private key for acces to ssh
        k = paramiko.RSAKey.from_private_key_file(os.environ['SSH_PRIVATE_KEY_PATH'])

        # Create an SSH client instance
        ssh_client = paramiko.SSHClient()

        # Automatically add untrusted hosts
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the SSH server
        ssh_client.connect(os.environ['SSH_HOSTNAME'], os.environ['SSH_PORT'], os.environ['SSH_USERNAME'],pkey=k)

        # Execute the command
        stdin, stdout, stderr = ssh_client.exec_command(command)

        # Print any errors
        print(stderr.read())

        # Read the output of the command
        output = stdout.read().decode()

        # Close the SSH connection
        ssh_client.close()

        return output
    
    except Exception as e:
        # Handle exceptions here
        print("An error occurred:", str(e))
        return None

