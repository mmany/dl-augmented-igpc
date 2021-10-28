volScalarField rAU(1.0/UEqn.A());
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U0, p0));
surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::flux(HbyA)
  + MRF.zeroFilter(fvc::interpolate(rAU)*fvc::ddtCorr(U0, phi0))
);

MRF.makeRelative(phiHbyA);

adjustPhi(phiHbyA, U0, p0);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p0, U0, phiHbyA, rAU, MRF);

// Non-orthogonal pressure corrector loop
while (piso.correctNonOrthogonal())
{
    // Pressure corrector

    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAU, p0) == fvc::div(phiHbyA)
    );

    pEqn.setReference(pRefCell, pRefValue);

    pEqn.solve(mesh.solver(p0.select(piso.finalInnerIter())));

    if (piso.finalNonOrthogonalIter())
    {
        phi0 = phiHbyA - pEqn.flux();
    }
}

#include "continuityErrs0.H"

U0 = HbyA - rAU*fvc::grad(p0);
U0.correctBoundaryConditions();
fvOptions.correct(U0);
