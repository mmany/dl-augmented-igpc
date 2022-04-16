// Solve the Momentum equation

MRF.correctBoundaryVelocity(U0);

fvVectorMatrix UEqn
(
    fvm::ddt(U0) 
    + fvm::div(phi0, U0)
    + fvc::div(phi1, U1)
    - nu0 * fvm::laplacian(U0)
    - nu1 * fvc::laplacian(U1)
    - fvm::laplacian(nut0,U0)
    - fvc::laplacian(nut1,U1)
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
