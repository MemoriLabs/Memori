from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "memori_python",
            path="core/bindings/python/Cargo.toml",
            binding=Binding.PyO3,
            py_limited_api=True,
        )
    ],
    zip_safe=False,
)
