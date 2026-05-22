#!/bin/sh
set -eu

CONFIG_PATH="/data/homeserver.yaml"
LOG_PATH="/homeserver.log"

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

# Synapse's generated log config writes to /homeserver.log. Point that file at
# the container log stream so compose logs capture startup and runtime output.
rm -f "${LOG_PATH}"
ln -s /proc/1/fd/1 "${LOG_PATH}"

exec python -m synapse.app.homeserver --config-path "${CONFIG_PATH}"
