# Third-party notices

This project implements Extreme Deconvolution and retains/refactors a prototype
whose exact historical inputs are no longer known. The maintainer reports that
the prototype was probably adapted from or informed by both the astroML XDGMM
implementation and the original `jobovy/extreme-deconvolution` software.
Accordingly, this project carries both notices conservatively.

The revisions below are the immutable sources inspected during the 2026-08-25
engineering provenance audit. They are custody and comparison references; they
are not claims that these exact revisions were the historical inputs used to
generate the prototype.

## astroML XDGMM

- Project: <https://github.com/astroML/astroML>
- Inspected revision: `d7485a8acd8d01d927fc381e3a58653e6cd47865`
- Inspected implementation:
  `astroML/density_estimation/xdeconv.py`
- Implementation SHA-256:
  `06a533b339065967294929e19c4e1359981642fee39f5f5ad81eda1a62f3d315`
- License source: `LICENSE.rst`
- License SHA-256:
  `829eccd5a3dc1dafa02fdfe6b810ff7a8d7c0dc97630eb3658d3cb8900e55384`

The following license text is reproduced verbatim from that pinned source:

```text
Copyright (c) 2012-2013, Jacob Vanderplas
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Original extreme-deconvolution software

- Project: <https://github.com/jobovy/extreme-deconvolution>
- Inspected revision: `a8a5988d2ab3ceeecbe7f0c23e0554d8a3a4222c`
- Inspected core implementation: `src/proj_EM_step.c`
- Core SHA-256:
  `53b37ca9a1baca8908e1f64c25ce8ef622a380bdd93f608a8274d1d6d4fe9d10`
- Inspected Python wrapper: `py/extreme_deconvolution.py`
- Wrapper SHA-256:
  `0340e31ab4d3fd2652cbf847c61e6c36888add630a5c151fefd0d226bcb07a49`
- License source: `LICENSE`
- License SHA-256:
  `e52808797a9bd901b30bbd0a42d2189090f9390803fa8102c1afbb9919f3c18e`

The following license text is reproduced verbatim from that pinned source:

```text
Copyright (c) 2008-2014, Jo Bovy, David W. Hogg, & Sam Roweis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: 

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. The name of the author may not be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Scholarly attribution

The Extreme Deconvolution method is described by Jo Bovy, David W. Hogg, and
Sam T. Roweis, “Extreme deconvolution: Inferring complete distribution
functions from noisy, heterogeneous and incomplete observations,” *The Annals
of Applied Statistics* 5(2B), 1657–1677 (2011),
<https://doi.org/10.1214/10-AOAS439>.

These notices do not imply that the upstream authors endorse this project or
authored its JAX-specific changes.
