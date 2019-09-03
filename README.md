# dippykit - A Digital Image Processing Library

Authors: Brighton Ancelin, [Motaz Alfarraj](https://motaz.me), [Ghassan AlRegib](https://ghassanalregib.info). 

This library was developed for the Georgia Tech's graduate course [ECE 6258: Digital Image Processing](https://ghassanalregib.info/ece6258) with Professor [Ghassan AlRegib](https://ghassanalregib.info).

Documentation can be found [here](https://dippykit.github.io/dippykit/).

## Purpose 

This package is intended for use by Digital Image Processing students. It serves as an educational resource.
The general methodology for the package is that each module contains functions pertinent to a specific concept or field.
For convenience to the user, all functions are available directly through the package itself. In other words, users are
not tasked with specifying modules to access specific members. The organization of modules is a logical "under-the-hood"
aspect in this regard.

## Known Issues

On some Mac devices this library has issues when running on Python 3.6.9 and up. Notably, Python may crash whenever one 
attempts to invoke the backend to matplotlib through functions such as ``dippykit.show``. We are working to resolve 
these issues, but for the moment we recommend any Mac users to use Python 3.6.8 with this library.

## Versioning Scheme
This code is maintained under the **Semantic Versioning 2.0.0** versioning scheme, further defined
[here](https://semver.org/) In essence, the version is represented as MAJOR.MINOR.PATCH and each increments by the
following rules:

* Increment MAJOR when incompatible API changes are made
* Increment MINOR when backwards-compatible functionality is added
* Increment PATCH when backwards-compatible bug fixes are made
