#!/bin/sh
set -eu

CONFIG_PATH="/data/homeserver.yaml"

if [ ! -f "${CONFIG_PATH}" ]; then
  python -m synapse.app.homeserver \
    --server-name localhost \
    --config-path "${CONFIG_PATH}" \
    --generate-config \
    --report-stats=no

  python3 - <<'PYEOF'
import yaml

with open("/data/homeserver.yaml") as f:
    cfg = yaml.safe_load(f)

for listener in cfg.get("listeners", []):
    if listener.get("port") == 8008:
        listener["bind_addresses"] = ["0.0.0.0"]

cfg["enable_registration"] = True
cfg["enable_registration_without_verification"] = True
cfg["suppress_key_server_warning"] = True
cfg["allow_public_rooms_without_auth"] = True
cfg["allow_public_rooms_over_federation"] = False

with open("/data/homeserver.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PYEOF
fi

# Keep Synapse in the foreground so the container stays alive even if a
# generated config defaults to daemon mode.
python3 - <<'PYEOF'
import yaml

with open("/data/homeserver.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["daemonize"] = False
with open("/data/homeserver.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PYEOF

LOG_CONFIG_PATH="$(python3 -c "
import yaml
cfg = yaml.safe_load(open('/data/homeserver.yaml'))
print(cfg.get('log_config', '/data/localhost.log.config'))
")"

# Replace Synapse's generated file logger with a console logger so startup and
# runtime logs are visible in docker compose logs across local Docker and DIND.
cat > "${LOG_CONFIG_PATH}" <<'EOF'
version: 1
disable_existing_loggers: false
formatters:
  generic:
    format: '%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: generic
    stream: ext://sys.stdout
root:
  level: INFO
  handlers: [console]
EOF

exec python -m synapse.app.homeserver --config-path "${CONFIG_PATH}"
