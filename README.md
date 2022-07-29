<img src="https://github.com/tizian/differential-mc-estimators/raw/master/docs/images/teaser.jpg" alt="Teaser">

# Monte Carlo Estimators for Differential Light Transport

Source code of the paper ["Monte Carlo Estimators for Differential Light Transport"](http://rgl.epfl.ch/publications/Zeltner2021MonteCarlo) by [Tizian Zeltner](https://tizianzeltner.com/), [Sébastien Speierer](http://rgl.epfl.ch/people/speierers), [Iliyan Georgiev](http://www.iliyan.com/), and [Wenzel Jakob](http://rgl.epfl.ch/people/wjakob) from SIGGRAPH 2021.

The implementation is based on the Mitsuba 3 renderer, see the lower part of the README.

## Compiling

Please see the [documentation](https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html) for details on how to compile this codebase.


## Results

The `results` directory contains a set of Jupyter notebooks that reproduce some of the experiments from the paper.

1. **Differential Microfacet Sampling Strategy**

   This notebook creates a list of plots to illustrate the differential Microfacet sampling strategies discusses in Section 4.1 of the paper.
   
2. **1D Reparameterizations**

   Simple 1D examples of the reparameterization ideas dicussed in Section 5 of the paper. Illustrates both the cases of ...
      - Parameter-dependent discontinuities from moving geometry.
      - Parameter-dependent discontinuities from static geometry in presence of attached sampling strategies.

3. **Estimator comparison (simple)**

   Forward and reverse mode differentiation of a simple microfacet material's roughness parameter. Compares a long list of differential estimators:
      - Detached emitter sampling
      - Detached BSDF sampling
      - Attached BSDF sampling
      
      - Detached MIS weights, detached BSDF sampling, detached emitter sampling
      - Attached MIS weights, attached BSDF sampling, detached emitter sampling
      - Detached MIS weights, attached BSDF sampling, detached emitter sampling (biased!)
      - Attached MIS weights, detached BSDF sampling, detached emitter sampling

      - Detached BSDF sampling using the differential microfacet sampling strategy
      - Detached MIS weights, detached differential microfacet BSDF sampling, detached emitter sampling

4. **Estimator comparison (roughness optimization)**
   
   A simple gradient based optimization recovering a roughness texture from a given reference rendering. Compares three different BSDF sampling techniques under two different illumination conditions. (Similar to Figure 12 from the paper):
   
      - Detached BSDF sampling
      - Attached BSDF sampling
      - Detached BSDF sampling using the differential microfacet sampling strategy

5. **Estimator comparison (Veach)**

   Reproduces the (forward mode) gradient images of the classical Veach test scene from Figure 11 in the paper. Compared to the simpler test scenes above, this scene contains (static) visibility discontinuities that causes naïve attached estimators to be biased. In addition to the previously listed estimators, two additional ones involving reparameterized attached sampling are used:
   
      - Reparameterized attached BSDF sampling
      - Attached MIS weights, reparameterized attached BSDF sampling, detached emitter sampling

6. **Estimator comparison (attached reparameterized)**

   Again illustrates the bias of attached BSDF sampling in presence of static visibility discontinuities, this time visualizing parameter gradient textures produced by reverse mode differentiation. Based on Figure 10 of the paper.
   
   Additionally includes a second test scene that shows the same issue for discontinous shading normals from a normal map texture.


<br><br>
<!-- <img src="https://github.com/mitsuba-renderer/mitsuba3/raw/master/docs/images/logo_plain.png" width="120" height="120" alt="Mitsuba logo"> -->

<!-- <img src="https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-data/master/docs/images/banners/banner_01.jpg"
alt="Mitsuba banner"> -->

# Mitsuba Renderer 3

| Documentation  | Tutorial videos  | Linux             | MacOS             | Windows           |
|      :---:     |      :---:       |       :---:       |       :---:       |       :---:       |
| [![docs][1]][2]| [![vids][9]][10] | [![rgl-ci][3]][4] | [![rgl-ci][5]][6] | [![rgl-ci][7]][8] |

[1]: https://readthedocs.org/projects/mitsuba/badge/?version=latest
[2]: https://mitsuba.readthedocs.io/en/latest/
[3]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:Mitsuba3_LinuxAmd64Clang10)/statusIcon.svg
[4]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=Mitsuba3_LinuxAmd64Clang10&guest=1
[5]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:Mitsuba3_LinuxAmd64gcc9)/statusIcon.svg
[6]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=Mitsuba3_LinuxAmd64gcc9&guest=1
[7]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:Mitsuba3_WindowsAmd64msvc2020)/statusIcon.svg
[8]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=Mitsuba3_WindowsAmd64msvc2020&guest=1
[9]: https://img.shields.io/badge/YouTube-View-green?style=plastic&logo=youtube
[10]: https://www.youtube.com/watch?v=9Ja9buZx0Cs&list=PLI9y-85z_Po6da-pyTNGTns2n4fhpbLe5&index=1

## Introduction

Mitsuba 3 is a research-oriented rendering system for forward and inverse light
transport simulation developed at [EPFL](https://www.epfl.ch) in Switzerland.
It consists of a core library and a set of plugins that implement functionality
ranging from materials and light sources to complete rendering algorithms. 

Mitsuba 3 is *retargetable*: this means that the underlying implementations and
data structures can transform to accomplish various different tasks. For
example, the same code can simulate both scalar (classic one-ray-at-a-time) RGB transport
or differential spectral transport on the GPU. This all builds on
[Dr.Jit](https://github.com/mitsuba-renderer/drjit), a specialized *just-in-time*
(JIT) compiler developed specifically for this project.

## Main Features

- **Cross-platform**: Mitsuba 3 has been tested on Linux (``x86_64``), macOS
  (``aarch64``, ``x86_64``), and Windows (``x86_64``).

- **High performance**: The underlying Dr.Jit compiler fuses rendering code
  into kernels that achieve state-of-the-art performance using
  an LLVM backend targeting the CPU and a CUDA/OptiX backend
  targeting NVIDIA GPUs with ray tracing hardware acceleration.

- **Python first**: Mitsuba 3 is deeply integrated with Python. Materials,
  textures, and even full rendering algorithms can be developed in Python,
  which the system JIT-compiles (and optionally differentiates) on the fly.
  This enables the experimentation needed for research in computer graphics and
  other disciplines.

- **Differentiation**: Mitsuba 3 is a differentiable renderer, meaning that it
  can compute derivatives of the entire simulation with respect to input
  parameters such as camera pose, geometry, BSDFs, textures, and volumes. It
  implements recent differentiable rendering algorithms developed at EPFL. 

- **Spectral & Polarization**: Mitsuba 3 can be used as a monochromatic
  renderer, RGB-based renderer, or spectral renderer. Each variant can
  optionally account for the effects of polarization if desired.

## Tutorial videos, documentation

We've recorded several [YouTube videos][10] that provide a gentle introduction
Mitsuba 3 and Dr.Jit. Beyond this you can find complete Juypter notebooks
covering a variety of applications, how-to guides, and reference documentation
on [readthedocs][2].

<!--
## Installation

We provide pre-compiled binary wheels via PyPI. Installing Mitsuba this way is as simple as running

```bash
pip install mitsuba
```

on the command line. The Python package includes four variants by default:

- ``scalar_spectral``
- ``scalar_rgb``
- ``llvm_ad_rgb``
- ``cuda_ad_rgb``

The first two perform classic one-ray-at-a-time simulation using either a RGB
or spectral color representation, while the latter two can be used for inverse
rendering on the CPU or GPU. To access additional variants, you will need to
compile a custom version of Dr.Jit using CMake. Please see the
[documentation](https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html)
for details on this.

### Requirements

- `Python >= 3.8`
- (optional) For computation on the GPU: `Nvidia driver >= 495.89`
- (optional) For vectorized / parallel computation on the CPU: `LLVM >= 11.1`

## Usage

Here is a simple "Hello World" example that shows how simple it is to render a
scene using Mitsuba 3 from Python:

```python
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
mi.set_variant('scalar_rgb')
# Load a scene
scene = mi.load_dict(mi.cornell_box())
# Render the scene
img = mi.render(scene)
# Write the rendered image to an EXR file
mi.Bitmap(img).write('cbox.exr')
```

Tutorials and example notebooks covering a variety of applications can be found
in the [documentation][2].
-->

## About

This project was created by [Wenzel Jakob](http://rgl.epfl.ch/people/wjakob).
Significant features and/or improvements to the code were contributed by
[Sébastien Speierer](https://speierers.github.io/),
[Nicolas Roussel](https://github.com/njroussel),
[Merlin Nimier-David](https://merlin.nimierdavid.fr/),
[Delio Vicini](https://dvicini.github.io/),
[Tizian Zeltner](https://tizianzeltner.com/),
[Baptiste Nicolet](https://bnicolet.com/),
[Miguel Crespo](https://mcrespo.me/),
[Vincent Leroy](https://github.com/leroyvn), and
[Ziyi Zhang](https://github.com/ziyi-zhang).
