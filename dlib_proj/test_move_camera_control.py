import requests
from requests.auth import HTTPDigestAuth


# Camera credentials and IP information
camera_ip = "172.16.14.10"  # Replace with your actual IP address
username = "admin"           # Replace with your username
password = "ZirRobotics"           # Replace with your password


# PTZ command to move the camera
# New send_ptz_command function using specific action codes
def send_ptz_command(code, channel=0, speed_vertical=1, speed_horizontal=1):
    try:
        # Adjusted PTZ control command with a specific code
        url = f"http://{camera_ip}/cgi-bin/ptz.cgi?action=start&channel={channel}&code={code}&arg1={speed_vertical}&arg2={speed_horizontal}&arg3=0"
        # Use digest authentication for the camera
        response = requests.get(url, auth=HTTPDigestAuth(username, password))

        if response.status_code == 200:
            print(f"Command '{code}' executed successfully.")
        else:
            print(f"Failed to execute '{code}'. Response code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending PTZ command: {e}")


# Test commands for specific PTZ actions
send_ptz_command(code="LeftUp")  # Attempt to tilt up with speed 5


# Stop movement
# send_ptz_command(command="stop", channel=0)
