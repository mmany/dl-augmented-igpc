# dl-augmented-igpc
A deep learning augmented solver in OpenFOAM built on top of the pisoFOAM solver, which uses a custom Inference Engine built with the PyTorch C++ Frontend (libtorch).
The Inference Engine currently supports only CPU operations, support for the use of the GPU for neural network inference is under development.

## About the solver
### Purpose and Theory
To be written.

## Installing dependencies
An official installation of OPENFOAM® is required to compile this new solver.
The solver has been tested on OpenFOAM v6.
The installation instructions of the OpenFOAM software from *The OpenFOAM Foundation* can be found in the [official release website](https://openfoam.org/version/6/).
OpenFOAM can be compiled from source for performance but an installation with Docker is usually sufficient. 

This solver relies on a custom inference engine which supports openfoam objects from the repository [mmany/openfoam-pytorch-inference](https://github.com/mmany/openfoam-pytorch-inference).
The inference engine requires the following dependencies:
..* the PyTorch [C++ Frontend](https://pytorch.org/) (libtorch, tested version 1.11.0)
..* the JSON library from the repository [nlohmann/json](https://github.com/nlohmann/json)
A local installation of these dependencies is to be performed and shortly explained in the following in a linux environment.

### libtorch
```shell
wget -q -O libtorch.zip  https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
unzip libtorch.zip
rm *.zip
```
### nlohmann-json
```shell
git clone https://github.com/nlohmann/json
```
## Compiling
Environment variables are defined, to ease the access to the files of the previously downloaded libraries.
```shell
export TORCH_PATH="path/to/libtorch"
export JSON_PATH="path/to/json"
```
Make sure not to add the last "/".
### Compiling with cmake

CMake is a cross platform compiling framework that can be used to compile a C++ project.
In this solver the compiling process with the different libraries with *cmake* is more complex than its *wmake* counterpart.
The CMakeLists.txt file is provided for informational purposes and compiling with cmake with the CMakeLists.txt file fails at linking stage.
Any help would be appreciated !

### Compiling with wmake
** Tested for Openfoam-6 on a linux 64 platform. **
OpenFOAM offers its own make command *wmake*.
OpenFOAM distributions prior to OpenFOAM-9 are compiled using the C++11 standard, whereas libtorch compiles with the C++14 standard.
The compiling of OpenFOAM solvers using the C++14 standard is usually discouraged, and the whole OpenFOAM distribution should be compiled again from source using the C++14 standard.
However, compiling with C++14 doesn't seem to yield any errors.

We configure the wmake command to compile using the C++14 standard, by changing the compiling rules.
```shell
sed -i "s/-std=c++11/-std=c++14/g" ${FOAM_PATH}/wmake/rules/linux64Gcc/c++
```

Cloning the git repo and compiling:
```shell
git clone --recursive https://github.com/mmany/dl-augmented-igpc
cd dl-augmented-igpc
wmake

#Optional: sourcing openfoam in order to call the newly compiled solver bin
source $FOAM_PATH/etc/bashrc
```
## Testing
A test case is provided under the [test](./test) folder. The test case can be run with the following commands:
```shell
blockMesh
pisoFoamPC_CNN
```
The results are visualized with Paraview using:
```shell
paraFoam
```
Inside Docker, Paraview is particularly slow. Paraview can be run outside Docker by creating a \*.foam (empty) file at the root of the test case folder and  by opening it from Paraview.

## To Do

- [ ] Installation Process with cmake
- [x] Installation Process with wmake
- [x] Link dependency with libtorch and nlohmann-json
- [x] Test case
- [x] Linking with Inference engine as a submodule

## Get in touch

If you would like to contribute, modify the code, or test compiling in different platforms, add new features, please create a pull request.

If you encounter any issues during compiling or testing, please kindly report it using the [issue tracker](https://github.com/mmany/dl-augmented-igpc/issues).

Feel free to contact me directly on my email to say hi, or to tell me if this is useful for another project, it always feels nice to know that the work is useful.

Contact: maheindrane.many@gmail.com
Maheindrane Many
M.Sc. in Aerospace
Technical University of Munich
