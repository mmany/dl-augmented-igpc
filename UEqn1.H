// Solve the Momentum equation

MRF.correctBoundaryVelocity(U1);

fvVectorMatrix UEqn
(
    fvm::ddt(U1) 
    + fvm::div(phi0, U1)
    + fvc::div(phi1, U0)
    - turbulence0->nu() * fvm::laplacian(U1)
    - turbulence1->nu() * fvc::laplacian(U0)
    - fvm::laplacian(nut0_cnn,U1)
    - fvc::laplacian(nut1_cnn,U0)
    //+ MRF.DDt(U1)
//==
    //fvOptions(U1)
);

UEqn.relax();

fvOptions.constrain(UEqn);
if (piso.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p1));

    fvOptions.correct(U1);
}
