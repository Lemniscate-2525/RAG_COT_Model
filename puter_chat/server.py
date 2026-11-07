"""Local HTTP server for the Puter.js AI Chat App.

Satisfies Puter.js security requirement:
'An application using Puter.js should be served by an HTTP server (not a file:// URL) to run properly.'
"""

import http.server
import socketserver
import os
import webbrowser
import sys

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Clean logging
        sys.stdout.write(f"[Puter HTTP Server] {args[0]} - {args[1]}\n")

def main():
    port = PORT
    for attempt in range(10):
        try:
            with socketserver.TCPServer(("", port), Handler) as httpd:
                url = f"http://localhost:{port}"
                print("=" * 60)
                print(f"🚀 Puter.js AI Chat App is running at: {url}")
                print("=" * 60)
                print("• Powered by Puter.js (Zero API keys required)")
                print("• Press Ctrl+C to stop the server\n")
                
                # Open in user browser automatically
                try:
                    webbrowser.open(url)
                except Exception:
                    pass

                httpd.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e) or e.errno in (98, 10048):
                port += 1
            else:
                raise e

if __name__ == "__main__":
    main()
