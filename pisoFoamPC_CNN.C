/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    pisoFoamPC

Description
    Transient solver for incompressible, turbulent flow, using the PISO
    algorithm and integration of the intrusive generalized Polynomial Chaos algorithm.

    Sub-models include:
    - turbulence modelling, i.e. laminar, RAS or LES
    - run-time selectable MRF and finite volume options, e.g. explicit porosity

\*---------------------------------------------------------------------------*/
// For CNN
#include "InferenceEngine/InferenceEngine.h"
#include "messageStream.H"

#ifndef OPENFOAM_DEPENDENCY
#define OPENFOAM_DEPENDENCY
#include "fvCFD.H"
#endif

#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pisoControl.H"
#include "fvOptions.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "postProcess.H"

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"


    turbulence0->validate();
    turbulence1->validate();

    Info << "******************************" << endl;
    Info << "@@@@@ Initializing CNN @@@@@@@" << endl;
    Info << "******************************" << endl;

    // Reading CNN presets
    #include "cnn_config.H"

    string model_path = cnn_model_path; 
    string inorm_path = input_normalizer_path; 
    string onorm_path = output_normalizer_path; 

    // Creating an instance of CNN object
    InferenceEngine cnn_instance(model_path, inorm_path, onorm_path);
    torch::NoGradGuard no_grad_guard;

    float UR_factor = under_relaxation_factor;
    Info << "Initial Under-Relaxation factor set to " << UR_factor << endl;

    cnn_instance.set_inlet_nut(nut0_cnn,0.0096); // Setting a custom value at the inlet face centers
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
	//forAll(U0, celli){
	//	U0[celli].component(1) = 0;
	//	U1[celli].component(1) = 0;
	//}

        #include "CourantNo0.H"

        // Pressure-velocity PISO corrector
        {
            #include "UEqn0.H"

            // --- PISO loop
            while (piso.correct())
            {
                #include "pEqn0.H"
            }
        }

        //laminarTransport0.correct();
        //turbulence0->correct();
	//nut0 = turbulence0->nut();
	
        // Pressure-velocity PISO corrector
        {
            #include "UEqn1.H"

            // --- PISO loop
            while (piso.correct())
            {
                #include "pEqn1.H"
            }
        }

        //laminarTransport1.correct();
        //turbulence1->correct();
	//nut1 = turbulence1->nut();
	//if (count>=num_iter){
		torch::Tensor input = cnn_instance.convertToTensor(U0, U1);
		torch::Tensor output = cnn_instance.predict(input);
		output = cnn_instance.under_relaxation(under_relaxation_factor);
		cnn_instance.updateFoamFieldChannelFlow(output, nut0_cnn, nut1_cnn);
	//}


        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
	//count=count+1;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
