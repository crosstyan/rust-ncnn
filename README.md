# rust-ncnn
[![GitHub license](https://img.shields.io/badge/license-apache--2--Clause-brightgreen.svg)](./LICENSE)

Rust bindings fork from
[tpoisonooo/rust-ncnn](https://github.com/tpoisonooo/rust-ncnn) for
[ncnn](https://github.com/tencent/ncnn).

[Original README](./README.old.md).

A fork fixed some problems.

## TODO

- [x] Find ncnn in system (fixed path) instead of building every time.
- [ ] Use cmake file/pkg-config file to find ncnn.
- [ ] Rebind it with `cxx` (Not sure if it's a good idea).
- [ ] Write some helper function instead of using `c_api.h` (too limited).