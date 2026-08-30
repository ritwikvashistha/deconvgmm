# Pinned Linux/amd64 CPU environment for XD-GEN-REF-BOVY-001 generation.
# The Bovy source archive is never fetched or copied into this image.  It is
# supplied read-only at runtime and verified by the generator before extraction.
FROM --platform=linux/amd64 debian:bookworm-20260824-slim@sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867

ARG DEBIAN_FRONTEND=noninteractive

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    XDGMM_BOVY_REFERENCE_CONTAINER=1 \
    XDGMM_BOVY_BASE_AMD64_DIGEST=sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867 \
    XDGMM_BOVY_BASE_INDEX_DIGEST=sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 \
    XDGMM_BOVY_DEBIAN_SNAPSHOT=20260824T000000Z \
    XDGMM_BOVY_RUNTIME_NETWORK=none

RUN set -eux; \
    rm -f /etc/apt/sources.list.d/debian.sources; \
    printf '%s\n' \
      'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20260824T000000Z bookworm main' \
      'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20260824T000000Z bookworm-updates main' \
      'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20260824T000000Z bookworm-security main' \
      > /etc/apt/sources.list; \
    printf '%s\n' 'Acquire::Check-Valid-Until "false";' \
      > /etc/apt/apt.conf.d/99snapshot; \
    apt-get update; \
    apt-get install --yes --no-install-recommends \
      gcc=4:12.2.0-3 \
      make=4.3-4.1 \
      libgsl-dev=2.7.1+dfsg-5+deb12u1 \
      python3=3.11.2-1+b1 \
      python3-numpy=1:1.24.2-1+deb12u1; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir --mode=0755 /input /output

WORKDIR /workspace

ENTRYPOINT ["python3", "/workspace/scripts/generate_bovy_parity_fixture.py"]
