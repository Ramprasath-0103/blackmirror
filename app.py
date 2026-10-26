import argparse
import threading
import webbrowser

from web.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the BlackMirror web app.")
    parser.add_argument("--port", type=int, default=8000, help="port to serve on (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print(f"BlackMirror is running at {url}   (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    create_app().run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
