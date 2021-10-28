// Solve the Momentum equation

MRF.correctBoundaryVelocity(U0);

fvVectorMatrix UEqn
(
    fvm::ddt(U0) 
    + fvm::div(phi0, U0)
    + fvc::div(phi1, U1)
    - fvm::laplacian(turbulence0->nut() +turbulence0->nu(),U0)
    - fvc::laplacian(turbulence1->nut() +turbulence1->nu(),U1)
    - fvc::div((turbulence0->nut()+turbulence0->nu())*dev2(T(fvc::grad(U0))))
    - fvc::div((turbulence1->nut()+turbulence1->nu())*dev2(T(fvc::grad(U1))))
    + MRF.DDt(U0)
==
    fvOptions(U0)
);

UEqn.relax();

fvOptions.constrain(UEqn);

if (piso.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p0));

    fvOptions.correct(U0);
}
