#include "mex.h"
#include "model.h"
#include "state.h"
#include "errors.h"

Observation_model *assignModel(const mxArray *mx_model) {
    char *name = NULL;
    if (mxIsStruct(mx_model))
    {
        mxArray *tmp = mxGetField(mx_model, 0, "name");
        if (tmp != NULL && mxIsChar(tmp))
            name = mxArrayToString(tmp);
        else
            mexErrMsgIdAndTxt(missing_input,
                "If first argument is a structure, it needs a field 'name' that is a string");
    }
    else if (mxIsChar(mx_model))
        name = mxArrayToString(mx_model);
    else
        mexErrMsgIdAndTxt(missing_input,
            "First argement must be the name of a model or a structure with a field 'name'"
            "that contains the name of a model");
    if (name == NULL)
        mexErrMsgIdAndTxt(missing_input, "Error reading model name.");
    Observation_model *model = findModel(name);
    if (model == NULL)
        mexErrMsgIdAndTxt(invalid_input, "Model %s not available", name);
    return model;
}

void initialize_theta(Observation_model *model, const mxArray *mx_theta, Theta *theta)
{
    if (model->n_theta == 0) {
        // Check if structure input
        if (!mxIsStruct(mx_theta))
            mexErrMsgIdAndTxt(invalid_input,
                "Third argument must be a structure containing parameter values");

        // Check if nested structure
        mxArray *mx_theta_x = mxGetField(mx_theta, 0, "x");

        if( mx_theta_x != NULL )
            initializeThetax(mx_theta_x, theta->x);
        else
            initializeThetax(mx_theta, theta->x);
    }
    else {
        // Check structure input
        if( !mxIsStruct(mx_theta) )
            mexErrMsgIdAndTxt(invalid_input,
                "Third argument must be a structure containing parameter values");

        // Check nested structure
        mxArray *mx_theta_x = mxGetField(mx_theta, 0, "x");
        mxArray *mx_theta_y = mxGetField(mx_theta, 0, "y");

        if( mx_theta_x == NULL )
            mexErrMsgIdAndTxt(invalid_input,
                "Third argument must have a field 'x' for this model.");

        if( mx_theta_y == NULL )
            mexErrMsgIdAndTxt(invalid_input,
                "Third argument must have a field 'y' for this model.");

        // Read state and model parameters
        initializeThetax(mx_theta_x, theta->x);
        theta->y = theta_y_alloc(model);
        initialize_theta_y(model, mx_theta_y, theta->y);
    }
}

void initialize_data(Observation_model *model, const mxArray *mx_data, Data *data)
{
    if (mxIsStruct(mx_data))
    {
        mxArray *mx_data_y = mxGetField(mx_data, 0, "y");

        if (mx_data_y == NULL)
            mexErrMsgIdAndTxt(missing_input,
                "If second argument is a structure, it must have a field 'y' for data.");

        if (!mxIsDouble(mx_data_y))
            mexErrMsgIdAndTxt(invalid_input,
                "Field 'y' of second argument must be double precision floating point");

        if (mxGetN(mx_data_y) != 1)
            mexErrMsgIdAndTxt(invalid_input, "Field 'y' of second argument must be a column vector");

        data->n = mxGetM(mx_data_y);                                                        
        data->m = mxGetM(mx_data_y);
        data->y = mxGetDoubles(mx_data_y);
    }
    else
    {
        if (!mxIsDouble(mx_data) && mxGetN(mx_data) != 1)
            mexErrMsgIdAndTxt( "mhessian:hessianMethod:invalidInputs",
                "Column vector of double required.");

        data->n = mxGetM(mx_data);
        data->m = mxGetM(mx_data);
        data->y = mxGetDoubles(mx_data);
    }
}

void initialize_theta_y(Observation_model *model, const mxArray *mx_theta_y, Theta_y *theta_y)
{
    for (int i_theta=0; i_theta<model->n_theta; i_theta++) {

        // Find and verify Matlab field for parameter i_theta
        Theta_y_constraints *theta_y_i_constraints = model->theta_y_constraints + i_theta;
        char *name = theta_y_i_constraints->name;
        mxArray *mx_theta_y_i = mxGetField(mx_theta_y, 0, theta_y_i_constraints->name);
        if (mx_theta_y_i == NULL)
            mexErrMsgIdAndTxt(invalid_input,
                "field 'y' of third argument must have a field %s for this model", name);

        // Find and verify dimension information
        int row_dim_index = theta_y_i_constraints->row_dimension_index;
        int col_dim_index = theta_y_i_constraints->col_dimension_index;
        int mx_n_rows = mxGetM(mx_theta_y_i);
        int mx_n_cols = mxGetN(mx_theta_y_i);

        // Consistency of number of rows with previous dimension information
        if (row_dim_index == -1) // Indicates Scalar
            theta_y->matrix[i_theta].n_rows = 1;
        else if (theta_y->dimension_parameters[row_dim_index] > 0)
            // Dimension constrained by previous matrix
            theta_y->matrix[i_theta].n_rows = theta_y->dimension_parameters[row_dim_index];
        else { // Dimension uncontrained, value stored for future comparisons
            theta_y->matrix[i_theta].n_rows = mx_n_rows;
            theta_y->dimension_parameters[row_dim_index] = mx_n_rows;
        }
        if (theta_y->matrix[i_theta].n_rows != mx_n_rows)
            mexErrMsgIdAndTxt(invalid_input,
                "Number of rows of field %s does not agree with the "
                "dimension of another parameter", name);

        // Consistency of number of columns with previous dimension information
        if (col_dim_index == -1) // Indicates Scalar
            theta_y->matrix[i_theta].n_cols = 1;
        else if (theta_y->dimension_parameters[col_dim_index] > 0)
            // Dimension constrained by previous matrix
            theta_y->matrix[i_theta].n_cols = theta_y->dimension_parameters[col_dim_index];
        else { // Dimension uncontrained, value stored for future comparisons
            theta_y->matrix[i_theta].n_cols = mx_n_cols;
            theta_y->dimension_parameters[col_dim_index] = mx_n_cols;
        }
        if (theta_y->matrix[i_theta].n_cols != mx_n_cols)
            mexErrMsgIdAndTxt(invalid_input,
                "Number of columns of field %s does not agree with the "
                "dimension of another parameter.", name);

        // Values are double
        if (!mxIsDouble(mx_theta_y_i))
            mexErrMsgIdAndTxt(invalid_input,
                "Parameter %s must be a double precision floating point values required.", name);

        theta_y->matrix[i_theta].p = mxGetDoubles(mx_theta_y_i);
        if ((theta_y_i_constraints->verify != NULL) && !theta_y_i_constraints->verify(theta_y->matrix + i_theta))
            mexErrMsgIdAndTxt(invalid_input, "Value of parameter %s is out of range", name);
    }
}
