volScalarField rAU(1.0/UEqn.A());
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U1, p1));
surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::flux(HbyA)
  + MRF.zeroFilter(fvc::interpolate(rAU)*fvc::ddtCorr(U1, phi1))
);

MRF.makeRelative(phiHbyA);

adjustPhi(phiHbyA, U1, p1);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p1, U1, phiHbyA, rAU, MRF);

// Non-orthogonal pressure corrector loop
while (piso.correctNonOrthogonal())
{
    // Pressure corrector

    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAU, p1) == fvc::div(phiHbyA)
    );

    pEqn.setReference(pRefCell, pRefValue);

    pEqn.solve(mesh.solver(p1.select(piso.finalInnerIter())));

    if (piso.finalNonOrthogonalIter())
    {
        phi1 = phiHbyA - pEqn.flux();
    }
}

#include "continuityErrs1.H"

U1 = HbyA - rAU*fvc::grad(p1);
U1.correctBoundaryConditions();
fvOptions.correct(U1);
