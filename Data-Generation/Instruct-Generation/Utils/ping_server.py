import socket
from concurrent.futures import ThreadPoolExecutor

# List of accessible servers
reachable_servers = []

# Function for checking whether the port is open
def is_port_open(ip, port=6000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # Timeout of 1 second
        try:
            # Attempts to establish a connection
            sock.connect((ip, port))
            return True
        except (socket.timeout, ConnectionError):
            return False

# Function for checking an endpoint
def check_endpoint(ip):
    base_url = f"http://{ip}:6000/v1/chat/completions"
    if is_port_open(ip):
        return base_url
    return None

# Main function for parallel processing
def main():
    base_ip_template = "0.0.0.{}" # Set base IP here
    ip_addresses = [base_ip_template.format(x) for x in range(1, 101)]  # All possible IPs

    # Use of a ThreadPoolExecutor for parallel port scans
    with ThreadPoolExecutor(max_workers=100) as executor:
        results = list(executor.map(check_endpoint, ip_addresses))

    # Filter the accessible servers
    global reachable_servers
    reachable_servers = [endpoint for endpoint in results if endpoint is not None]

    # Output results
    if reachable_servers:
        print("\nAccessible servers found:")
        for server in reachable_servers:
            print(server)
    else:
        print("No accessible servers found.")

# Start script
if __name__ == "__main__":
    main()
