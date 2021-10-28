// Solve the Momentum equation

MRF.correctBoundaryVelocity(U1);

fvVectorMatrix UEqn
(
    fvm::ddt(U1) 
    + fvm::div(phi0, U1)
    + fvc::div(phi1, U0)
    - fvm::laplacian(turbulence0->nut() +turbulence0->nu(),U1)
    - fvc::laplacian(turbulence1->nut() +turbulence1->nu(),U0)
    - fvc::div((turbulence0->nut()+turbulence0->nu())*dev2(T(fvc::grad(U1))))
    - fvc::div((turbulence1->nut()+turbulence1->nu())*dev2(T(fvc::grad(U0))))
    + MRF.DDt(U1)
==
    fvOptions(U1)
);

UEqn.relax();

fvOptions.constrain(UEqn);
if (piso.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p1));

    fvOptions.correct(U1);
}
