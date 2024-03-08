# Adds ngrok ingress, to use add `--extension ngrok` to the command line options
#
# Parameters can be customized in settings.json of webui, e.g.:
# {"ngrok": {"basic_auth":"user:password"} }
# or
# {"ngrok": {"oauth_provider":"google", "oauth_allow_emails":["asdf@asdf.com"]} }
#
# See this example for full list of options: https://github.com/ngrok/ngrok-py/blob/main/examples/ngrok-connect-full.py
# or the README.md in this directory.

from modules import shared

# Pick up host/port command line arguments
host = shared.args.listen_host if shared.args.listen_host and shared.args.listen else '127.0.0.1'
port = shared.args.listen_port if shared.args.listen_port else '7860'

# Default options
options = {
    'addr': f"{host}:{port}",
    'authtoken_from_env': True,
    'session_metadata': 'text-generation-webui',
}


def ui():
    settings = shared.settings.get("ngrok")
    if settings:
        options.update(settings)

    try:
        import ngrok
        tunnel = ngrok.connect(**options)
        shared.logger.info(f"Ingress được thiết lập tại: {tunnel.url()}")
    except ModuleNotFoundError:
        shared.logger.error("===> không tìm thấy thư viện ngrok, vui lòng chạy `pip install -r extensions/ngrok/requirements.txt`")