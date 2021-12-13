
#include "fileName.H"
IOdictionary cnn_config
(
IOobject
(
"cnn_config", /// The dictionary file
runTime.constant(), /// Relative path (from case root)
mesh, /// The Time object
IOobject::MUST_READ, /// Read for constructor
IOobject::NO_WRITE /// Foam::Time writeControl
)
);

fileName cnn_model_path(cnn_config.lookup("model_path"));
fileName input_normalizer_path(cnn_config.lookup("input_normalizer_path"));
fileName output_normalizer_path(cnn_config.lookup("output_normalizer_path"));
scalar under_relaxation_factor(readScalar(cnn_config.lookup("under_relaxation_factor")));

