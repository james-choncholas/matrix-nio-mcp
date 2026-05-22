#!/bin/sh
set -eu

CONFIG_PATH="/data/homeserver.yaml"
if [ ! -f "${CONFIG_PATH}" ]; then
  python -m synapse.app.homeserver \
    --server-name localhost \
    --config-path "${CONFIG_PATH}" \
    --generate-config \
    --report-stats=no

  sed -i '/^  - bind_addresses:/,/^    port: 8008$/c\
  - bind_addresses:\
    - 0.0.0.0\
    port: 8008' "${CONFIG_PATH}"

  if grep -q '^enable_registration:' "${CONFIG_PATH}"; then
    sed -i 's/^enable_registration: false$/enable_registration: true/' "${CONFIG_PATH}"
  else
    printf '\nenable_registration: true\n' >> "${CONFIG_PATH}"
  fi

  cat >> "${CONFIG_PATH}" <<'EOF'
enable_registration_without_verification: true
suppress_key_server_warning: true
allow_public_rooms_without_auth: true
allow_public_rooms_over_federation: false
EOF
fi

# Keep Synapse in the foreground so the container stays alive even if a
# generated config defaults to daemon mode.
if grep -q '^daemonize:' "${CONFIG_PATH}"; then
  sed -i 's/^daemonize: .*/daemonize: false/' "${CONFIG_PATH}"
else
  printf '\ndaemonize: false\n' >> "${CONFIG_PATH}"
fi

LOG_CONFIG_PATH="$(awk -F': ' '/^log_config:/ {gsub(/"/, "", $2); print $2; exit}' "${CONFIG_PATH}")"
if [ -z "${LOG_CONFIG_PATH}" ]; then
  LOG_CONFIG_PATH="/data/localhost.log.config"
fi

# Replace Synapse's generated file logger with a console logger so startup and
# runtime logs are visible in `docker compose logs` across local Docker and DIND.
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
