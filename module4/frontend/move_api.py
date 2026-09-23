from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess, urllib.parse, os
DRIVE = os.path.expanduser("~/lab-m4/scripts/drive_turtle.sh")

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        q = urllib.parse.urlparse(self.path)
        if q.path != "/move":
            self.send_response(404); self.end_headers(); return
        cmd = urllib.parse.parse_qs(q.query).get("cmd", ["停"])[0]
        subprocess.run([DRIVE, cmd], timeout=30)
        self.send_response(200); self.end_headers()
        self.wfile.write(f"ok:{cmd}".encode())
    def log_message(self, *args):
        pass

if __name__ == "__main__":
    print("move API: http://127.0.0.1:5174/move?cmd=前进")
    HTTPServer(("127.0.0.1", 5174), H).serve_forever()
