Info<< "Reading molecular viscosity scalars\n" << endl;
IOdictionary transportProperties
(
  IOobject
   (
    "transportProperties", /// The dictionary file
    runTime.constant(), /// Relative path (from case root)
    mesh, /// The Time object
    IOobject::MUST_READ, /// Read for constructor
    IOobject::NO_WRITE /// Foam::Time writeControl
   )
);
dimensionedScalar nu0
(
    "nu0",
    dimViscosity,
    transportProperties.lookup("nu0")
);
dimensionedScalar nu1
(
    "nu1",
    dimViscosity,
    transportProperties.lookup("nu1")
);

Info<< "Reading field nut0\n" << endl;
volScalarField nut0
(
    IOobject
    (
        "nut0",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field nut1\n" << endl;
volScalarField nut1
(
    IOobject
    (
        "nut1",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);



Info<< "Reading field p0\n" << endl;
volScalarField p0
(
    IOobject
    (
        "p0",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field p1\n" << endl;
volScalarField p1
(
    IOobject
    (
        "p1",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U0\n" << endl;
volVectorField U0
(
    IOobject
    (
        "U0",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U1\n" << endl;
volVectorField U1
(
    IOobject
    (
        "U1",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field phi0" <<endl;
surfaceScalarField phi0
(
    IOobject
    (
        "phi0",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    fvc::flux(U0)
);

Info<< "Reading field phi1" <<endl;
surfaceScalarField phi1
(
    IOobject
    (
        "phi1",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    fvc::flux(U1)
);
//#include "createPhi.H"
label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p0, piso.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p0.name());
setRefCell(p1, piso.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p1.name());

#include "createMRF.H"
#include "createFvOptions.H"

//singlePhaseTransportModel laminarTransport0(U0, phi0, "nu0");
//singlePhaseTransportModel laminarTransport1(U1, phi1, "nu1");
//
//autoPtr<incompressible::turbulenceModel> turbulence0
//(
//    incompressible::turbulenceModel::New(U0, phi0, laminarTransport0, "turbulenceProperties",0)
//);
//autoPtr<incompressible::turbulenceModel> turbulence1
//(
//    incompressible::turbulenceModel::New(U1, phi1, laminarTransport1, "turbulenceProperties",1)
//);
///*
//volScalarField nut0
// (
//        IOobject
//         (
//             "nut0",
//             runTime.timeName(),
//             mesh,
//             IOobject::NO_READ,
//            IOobject::AUTO_WRITE
//        ),
//        turbulence0->nut()
// );
//
//volScalarField nut1
// (
//        IOobject
//         (
//             "nut1",
//             runTime.timeName(),
//             mesh,
//             IOobject::NO_READ,
//            IOobject::AUTO_WRITE
//        ),
//        turbulence1->nut()
// );
//*/
