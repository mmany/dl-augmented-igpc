// Solve the Momentum equation

MRF.correctBoundaryVelocity(U0);

fvVectorMatrix UEqn
(
    fvm::ddt(U0) 
    + fvm::div(phi0, U0)
    + fvc::div(phi1, U1)
    - turbulence0->nu() * fvm::laplacian(U0)
    - turbulence1->nu() * fvc::laplacian(U1)
    //- fvm::laplacian(turbulence0->nut(),U0)
    //- fvc::laplacian(turbulence1->nut(),U1)
    //+ MRF.DDt(U0)
//==
    //fvOptions(U0)
);

UEqn.relax();

fvOptions.constrain(UEqn);

if (piso.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p0));

    fvOptions.correct(U0);
}
