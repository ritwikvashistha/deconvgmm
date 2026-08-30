#!/usr/bin/env bash
# Build the pinned reference environment, then generate with networking off.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 /path/to/pinned-source.tar.gz /path/to/output-directory" >&2
  exit 2
fi

SCRIPT_DIRECTORY_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT_PATH="$(cd "${SCRIPT_DIRECTORY_PATH}/../../.." && pwd -P)"
SOURCE_ARCHIVE_PATH="$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")"
CONTAINER_ENGINE_COMMAND="${XDGMM_CONTAINER_ENGINE:-docker}"
REFERENCE_IMAGE_TAG="xdgmm-bovy-reference:a8a5988d2ab3ceeecbe7f0c23e0554d8a3a4222c"

if [[ ! -f "${SOURCE_ARCHIVE_PATH}" ]]; then
  echo "source archive is not a regular file: ${SOURCE_ARCHIVE_PATH}" >&2
  exit 2
fi
mkdir -p "$2"
OUTPUT_DIRECTORY_PATH="$(cd "$2" && pwd -P)"

"${CONTAINER_ENGINE_COMMAND}" build \
  --platform linux/amd64 \
  --file "${SCRIPT_DIRECTORY_PATH}/Dockerfile" \
  --tag "${REFERENCE_IMAGE_TAG}" \
  "${SCRIPT_DIRECTORY_PATH}"

REFERENCE_IMAGE_ID="$(${CONTAINER_ENGINE_COMMAND} image inspect --format '{{.Id}}' "${REFERENCE_IMAGE_TAG}")"
if [[ "${REFERENCE_IMAGE_ID}" != sha256:* ]]; then
  echo "container engine returned an unexpected image ID: ${REFERENCE_IMAGE_ID}" >&2
  exit 2
fi

"${CONTAINER_ENGINE_COMMAND}" run \
  --rm \
  --platform linux/amd64 \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges=true \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,mode=1777,size=256m \
  --mount "type=bind,src=${SOURCE_ARCHIVE_PATH},dst=/input/source.tar.gz,readonly" \
  --mount "type=bind,src=${PROJECT_ROOT_PATH},dst=/workspace,readonly" \
  --mount "type=bind,src=${OUTPUT_DIRECTORY_PATH},dst=/output" \
  --env "XDGMM_BOVY_CONTAINER_IMAGE_ID=${REFERENCE_IMAGE_ID}" \
  "${REFERENCE_IMAGE_TAG}" \
  --source-archive /input/source.tar.gz \
  --output /output/bovy_general_ref_001.npz \
  --metadata-output /output/bovy_general_ref_001.metadata.json
